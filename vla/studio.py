"""Recording studio: a local web page to watch the cameras and run data collection.

    ./vla/studio.sh single        # then open http://localhost:8765
    ./vla/studio.sh dual

The page shows the live camera feeds and joint positions, and has buttons for connect, start,
save and discard an episode, and upload the dataset to the Hub. It records the same LeRobot
dataset as `lerobot-record` (same features, `save_episode`, `push_to_hub`), but it runs its own
control loop. That way the cameras are opened once, and the page can show them while the leader
drives the follower.

Buttons are also on the keyboard (in the page): Space start, Right arrow save, Left arrow discard.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import draccus
import numpy as np

# Imported for their side effect: registering the robot, teleop and camera types with draccus.
from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.robots import RobotConfig, bi_so_follower, make_robot_from_config, so_follower  # noqa: F401
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    make_teleoperator_from_config,
    so_leader,
)
from lerobot.utils.constants import ACTION, HF_LEROBOT_HOME, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts

PAGE = Path(__file__).with_name("studio.html")
log = logging.getLogger("studio")
ARMS = Path(__file__).with_name("arms.json")


def resolve_ports(cfg):
    """A copy of a robot/teleop config with every "serial:<n>" port replaced by that board's
    /dev/tty* now. Done on each connect, so arms plugged in after the page started are found."""
    import detect_arms  # same folder

    cfg = copy.deepcopy(cfg)
    names = {a["serial"]: a["id"] for a in json.loads(ARMS.read_text())["arms"]} if ARMS.exists() else {}
    now = {i["serial"]: p for p, i in detect_arms.ports().items()}
    for holder in (cfg, getattr(cfg, "left_arm_config", None), getattr(cfg, "right_arm_config", None)):
        port = getattr(holder, "port", None)
        if not (isinstance(port, str) and port.startswith("serial:")):
            continue
        serial = port.removeprefix("serial:")
        name = names.get(serial, f"arm with USB serial {serial}")
        if serial not in now:
            raise RuntimeError(f"{name} is not plugged in (follower: is its 12 V supply on?)")
        if not os.access(now[serial], os.R_OK | os.W_OK):
            import grp
            import getpass

            if getpass.getuser() in grp.getgrnam("dialout").gr_mem:  # joined after this session began
                raise RuntimeError(f"{name} is on {now[serial]}. You are in the dialout group, but the "
                                   "studio was started before that: stop it, run 'newgrp dialout' in "
                                   "that terminal (or log out and back in), and start it again")
            raise RuntimeError(f"{name} is on {now[serial]}, but you may not open it: "
                               "sudo usermod -aG dialout $USER, then log out and back in")
        holder.port = now[serial]
    return cfg


@dataclass
class Hardware:
    """The same --robot.* / --teleop.* flags lerobot-record takes (built by lib.sh)."""

    robot: RobotConfig
    teleop: TeleoperatorConfig


class Studio:
    """Owns the arms, cameras and dataset. One control thread; HTTP handlers only flip state.

    States: disconnected -> connecting -> preview (cameras only) | idle (teleop running)
            idle -> recording -> saving -> idle (-> next episode after the reset time if auto)
            idle -> uploading -> idle
    """

    def __init__(self, hw: Hardware, mode: str, repo_id: str, task: str, fps: int,
                 episode_time_s: float, reset_time_s: float, private: bool, tags: list[str],
                 views: list[str] | None = None, configured: dict[str, str] | None = None):
        self.hw, self.mode, self.repo_id, self.task, self.fps = hw, mode, repo_id, task, fps
        # Fixed camera slots on the page (config.sh CAMERA_VIEWS), and name -> arms for every camera
        # in CAMERA_TABLE, so an empty slot can say why it is empty.
        self.views = list(views or hw.robot.cameras)
        self.views += [c for c in hw.robot.cameras if c not in self.views]
        self.configured = configured if configured is not None else {c: "both" for c in hw.robot.cameras}
        self.cam_errors: dict[str, str] = {}
        self.episode_time_s, self.reset_time_s = episode_time_s, reset_time_s
        self.private, self.tags = private, tags
        self.root = HF_LEROBOT_HOME / repo_id

        self.lock = threading.RLock()
        self.state = "disconnected"
        self.message, self.error, self.url = "Not connected.", "", ""
        self.auto_next = False
        self.robot = self.teleop = self.dataset = self.features = None
        self.cams: dict = {}
        self.processors = None
        self.frames: dict[str, np.ndarray] = {}
        self.joints: dict[str, float] = {}
        self.hz = 0.0
        self.ep_start = 0.0
        self.ep_frames = 0
        self.reset_until = 0.0
        self.session_saved = 0
        self.ep_task = task
        self._thread: threading.Thread | None = None
        self._running = False
        self._shutting_down = False
        self._jpeg_cache: dict[str, tuple[np.ndarray, bytes]] = {}

    # ------------------------------------------------------------ status

    def episodes_on_disk(self) -> int:
        if self.dataset is not None:
            return self.dataset.num_episodes
        info = self.root / "meta" / "info.json"
        return json.loads(info.read_text()).get("total_episodes", 0) if info.exists() else 0

    def slots(self) -> list[dict]:
        """One entry per fixed camera slot: live, or why it is empty."""
        out = []
        for name in self.views:
            img = self.frames.get(name)
            if img is not None:
                out.append({"name": name, "live": True, "size": f"{img.shape[1]}x{img.shape[0]}"})
                continue
            if name not in self.configured:
                why = "Not set up yet: add its line to CAMERA_TABLE in vla/config.sh"
            elif name not in self.hw.robot.cameras:
                why = ("The other arm's wrist: used in dual-arm mode only"
                       if self.configured[name] in ("a", "b") else "Not used in this mode")
            elif name in self.cam_errors:
                why = self.cam_errors[name]
            elif self.state in ("connecting", "disconnecting"):
                why = "Connecting…"
            else:
                why = "Not connected"
            out.append({"name": name, "live": False, "why": why})
        return out

    def status(self) -> dict:
        now = time.monotonic()
        with self.lock:
            return {
                "slots": self.slots(),
                "state": self.state, "mode": self.mode, "repo_id": self.repo_id, "task": self.task,
                "fps": self.fps, "hz": round(self.hz, 1), "cameras": sorted(self.frames),
                "camera_sizes": {k: f"{v.shape[1]}x{v.shape[0]}" for k, v in self.frames.items()},
                "joints": {k: round(v, 1) for k, v in self.joints.items()},
                "episodes": self.episodes_on_disk(), "session_saved": self.session_saved,
                "elapsed": round(now - self.ep_start, 1) if self.state == "recording" else 0,
                "frames": self.ep_frames, "episode_time_s": self.episode_time_s,
                "reset_time_s": self.reset_time_s,
                "reset_left": round(max(0.0, self.reset_until - now), 1) if self.auto_next else 0,
                "auto_next": self.auto_next, "message": self.message, "error": self.error,
                "url": self.url, "private": self.private,
            }

    def jpeg(self, cam: str) -> bytes | None:
        """Latest frame as JPEG, encoded once per new frame however many viewers there are."""
        img = self.frames.get(cam)
        if img is None:
            return None
        cached = self._jpeg_cache.get(cam)
        if cached and cached[0] is img:
            return cached[1]
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return None
        self._jpeg_cache[cam] = (img, buf.tobytes())
        return self._jpeg_cache[cam][1]

    def _say(self, msg: str, error: str = "") -> None:
        self.message, self.error = msg, error
        (log.error if error else log.info)("%s %s", msg, error)

    # ------------------------------------------------------------ connect

    def connect(self, cameras_only: bool) -> None:
        with self.lock:
            ok = self.state == "disconnected" or (self.state == "preview" and not cameras_only)
            if not ok:
                return
            prev, self.state = self.state, "connecting"
        if prev == "preview":  # the robot opens the cameras itself, so release them first
            self._stop_loop()
            self._close_hardware()
        with self.lock:
            self._say("Connecting cameras..." if cameras_only else "Connecting arms and cameras...")
        threading.Thread(target=self._connect, args=(cameras_only,), daemon=True).start()

    def _connect(self, cameras_only: bool) -> None:
        try:
            self.cam_errors = {}
            if cameras_only:  # each camera on its own: one bad camera must not hide the others
                failed = []
                for name, cam in make_cameras_from_configs(self.hw.robot.cameras).items():
                    try:
                        cam.connect()
                        self.cams[name] = cam
                    except Exception as e:  # noqa: BLE001
                        self.cam_errors[name] = f"Did not open: {str(e).strip().splitlines()[0]}"
                        failed.append(f"{name}: {str(e).strip().splitlines()[0]}")
                if not self.cams:
                    raise RuntimeError("no camera opened. " + " | ".join(failed))
                state, msg = "preview", "Camera preview. Connect the arms to record."
                if failed:
                    msg = f"Preview without {len(failed)} camera(s): " + " | ".join(failed)
            else:
                robot_cfg, teleop_cfg = resolve_ports(self.hw.robot), resolve_ports(self.hw.teleop)
                self.teleop = make_teleoperator_from_config(teleop_cfg)
                self.robot = make_robot_from_config(robot_cfg)
                # Leader first, as lerobot-record does. calibrate=False: LeRobot's calibration waits
                # for Enter in the terminal, which would hang the page. These arms are calibrated by
                # the Hardware team and must never be recalibrated.
                for dev in (self.teleop, self.robot):
                    dev.connect(calibrate=False)
                    if not getattr(dev, "is_calibrated", True):
                        raise RuntimeError(
                            f"{dev}: its calibration file does not match what is stored in the servos. "
                            "rakesh_leader: expected once; run ./vla/teleop.sh single (SINGLE_ARM=a) and "
                            "press ENTER at 'use provided calibration file'. Any other arm: stop and ask "
                            "the Hardware team; never recalibrate (configs/calibration/README.md on hardware).")
                self.processors = make_default_processors()
                tap, _, rop = self.processors
                self.features = combine_feature_dicts(
                    aggregate_pipeline_dataset_features(
                        pipeline=tap, use_videos=True,
                        initial_features=create_initial_features(action=self.robot.action_features)),
                    aggregate_pipeline_dataset_features(
                        pipeline=rop, use_videos=True,
                        initial_features=create_initial_features(observation=self.robot.observation_features)),
                )
                state, msg = "idle", "Connected. The leader drives the follower. Press Start to record."
        except Exception as e:  # noqa: BLE001 - show any hardware error on the page
            self._close_hardware()
            with self.lock:
                self.state = "disconnected"
                self._say("Connect failed.", f"{type(e).__name__}: {e}")
            if not cameras_only:  # keep the camera view up while the arms are sorted out
                error = self.error
                self._connect(cameras_only=True)
                with self.lock:
                    if self.state == "preview":
                        self._say("Arms not connected; showing the cameras.", error)
            return
        with self.lock:
            self.state = state
            self._say(msg)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        with self.lock:
            if self.state in ("disconnected", "connecting", "saving", "uploading"):
                return
            if self.state == "recording":
                self.dataset.clear_episode_buffer()
            was_preview = self.state == "preview"
            self.state = "disconnecting"
        self._stop_loop()
        if self.dataset is not None:
            self.dataset.finalize()  # saved episodes stay on disk; upload resumes it later
            self.dataset = None
        self._close_hardware()
        with self.lock:
            self.frames, self.joints, self.hz = {}, {}, 0.0
            self.state = "disconnected"
            self._say("Disconnected. Saved episodes are on disk.")
        if not was_preview and not self._shutting_down:  # arms off: back to the camera preview
            self.connect(cameras_only=True)

    def _stop_loop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    def _close_hardware(self) -> None:
        for dev in (self.robot, self.teleop, *self.cams.values()):
            try:
                if dev is not None and dev.is_connected:
                    dev.disconnect()
            except Exception as e:  # noqa: BLE001
                log.warning("disconnect: %s", e)
        self.robot = self.teleop = None
        self.cams = {}

    # ------------------------------------------------------------ control loop

    def _loop(self) -> None:
        period = 1.0 / self.fps
        cam_keys = list(self.hw.robot.cameras)
        last, avg_dt = time.perf_counter(), period
        while self._running:
            t0 = time.perf_counter()
            try:
                if self.cams:
                    frames = dict(self.frames)
                    for k, c in self.cams.items():
                        try:
                            frames[k] = c.read_latest()
                        except Exception:  # noqa: BLE001 - keep the other cameras live
                            frames.pop(k, None)
                    self.frames = frames
                else:
                    self._step(cam_keys)
            except Exception as e:  # noqa: BLE001 - a bus or camera error should not kill the page
                with self.lock:
                    self._say("Control loop error (still running).", f"{type(e).__name__}: {e}")
                time.sleep(0.2)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
            now = time.perf_counter()
            avg_dt = 0.9 * avg_dt + 0.1 * (now - last)  # average the period, not 1/period
            self.hz, last = 1.0 / avg_dt, now

    def _step(self, cam_keys: list[str]) -> None:
        """One tick of lerobot-record's record_loop: observe, teleop, send, maybe record."""
        tap, rap, rop = self.processors
        obs = self.robot.get_observation()
        obs_p = rop(obs)
        act = self.teleop.get_action()
        act_p = tap((act, obs))
        self.robot.send_action(rap((act_p, obs)))

        self.frames = {k: obs[k] for k in cam_keys if k in obs}
        self.joints = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}

        with self.lock:
            if self.state == "recording":
                frame = {**build_dataset_frame(self.dataset.features, obs_p, prefix=OBS_STR),
                         **build_dataset_frame(self.dataset.features, act_p, prefix=ACTION),
                         "task": self.ep_task}
                self.dataset.add_frame(frame)
                self.ep_frames += 1
                if 0 < self.episode_time_s <= time.monotonic() - self.ep_start:  # 0: until Stop
                    self.save()
            elif self.state == "idle" and self.auto_next and self.reset_until and time.monotonic() >= self.reset_until:
                self.start()

    # ------------------------------------------------------------ episodes

    def _open_dataset(self) -> None:
        if self.dataset is not None:
            return
        n_cams = len(self.hw.robot.cameras)
        if (self.root / "meta" / "info.json").exists():
            ds = LeRobotDataset.resume(self.repo_id, root=self.root, image_writer_threads=4 * n_cams)
            try:  # same cameras, sizes, joints and fps as the episodes already recorded
                sanity_check_dataset_robot_compatibility(ds, self.robot, self.fps, self.features)
            except ValueError as e:
                ds.finalize()
                have = sorted(k.removeprefix("observation.images.") for k in ds.features
                              if k.startswith("observation.images."))
                raise RuntimeError(f"{self.repo_id} was recorded with cameras {have} and a different setup "
                                   f"than now ({e}). Use the same cameras, or record into a new dataset "
                                   "(DATASET_SINGLE / DATASET_DUAL in config.sh).") from None
            self.dataset = ds
        else:
            self.dataset = LeRobotDataset.create(self.repo_id, self.fps, root=self.root,
                                                 robot_type=self.robot.robot_type, features=self.features,
                                                 use_videos=True, image_writer_threads=4 * n_cams)

    def start(self, task: str | None = None) -> None:
        with self.lock:
            if self.state != "idle":
                return
            if task:
                self.task = task.strip()
            if not self.task:
                self._say("Type a task instruction first.", "empty task")
                return
            try:
                self._open_dataset()
            except Exception as e:  # noqa: BLE001
                self._say("Could not open the dataset.", f"{type(e).__name__}: {e}")
                return
            self.ep_task, self.ep_frames, self.reset_until = self.task, 0, 0.0
            self.ep_start = time.monotonic()
            self.state = "recording"
            self._say(f"Recording episode {self.dataset.num_episodes}.")

    def save(self) -> None:
        with self.lock:
            if self.state != "recording":
                return
            if self.ep_frames == 0:
                self.state = "idle"
                return
            self.state = "saving"
            self._say(f"Saving episode {self.dataset.num_episodes} ({self.ep_frames} frames)...")
        threading.Thread(target=self._save, daemon=True).start()

    def _save(self) -> None:
        try:
            self.dataset.save_episode()  # encodes the videos; the arm keeps following meanwhile
            with self.lock:
                self.session_saved += 1
                self.state = "idle"
                self.reset_until = time.monotonic() + self.reset_time_s
                nxt = f" Next starts in {self.reset_time_s:.0f} s." if self.auto_next else ""
                self._say(f"Saved. {self.dataset.num_episodes} episodes on disk. Reset the scene.{nxt}")
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.dataset.clear_episode_buffer()
                self.state = "idle"
                self._say("Saving failed; the episode was dropped.", f"{type(e).__name__}: {e}")

    def discard(self) -> None:
        with self.lock:
            if self.state != "recording":
                return
            self.dataset.clear_episode_buffer()
            self.state, self.reset_until = "idle", 0.0
            self._say("Episode discarded. Reset the scene and press Start to redo it.")

    def set_auto(self, on: bool) -> None:
        with self.lock:
            self.auto_next = bool(on)
            if not on:
                self.reset_until = 0.0

    def upload(self) -> None:
        with self.lock:
            if self.state != "idle":
                return
            if self.episodes_on_disk() == 0:
                self._say("Nothing to upload yet.", "no episodes")
                return
            self.state, self.auto_next, self.reset_until = "uploading", False, 0.0
            self._say(f"Uploading {self.repo_id} to the Hub...")
        threading.Thread(target=self._upload, daemon=True).start()

    def _upload(self) -> None:
        try:
            ds = self.dataset
            if ds is None:  # nothing recorded since connecting: reopen what is on disk
                ds = LeRobotDataset.resume(self.repo_id, root=self.root)
            ds.finalize()
            ds.push_to_hub(tags=self.tags, private=self.private)
            with self.lock:
                self.dataset = None  # finalized; the next Start resumes it
                self.url = f"https://huggingface.co/datasets/{self.repo_id}"
                self.state = "idle"
                self._say(f"Uploaded {ds.num_episodes} episodes.")
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.dataset = None
                self.state = "idle"
                self._say("Upload failed; episodes are still on disk.", f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------- HTTP


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet: the page polls status twice a second
            pass

        def _json(self, obj, code=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlsplit(self.path).path  # the page adds ?t=... to stream URLs to defeat caching
            if path in ("/", "/index.html"):
                body = PAGE.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/status":
                self._json(studio.status())
            elif path.startswith("/stream/"):
                self._stream(path.split("/")[2])
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _stream(self, cam: str):
            """MJPEG: the browser shows it in a plain <img>. About 15 fps, independent of the control loop."""
            if cam not in studio.frames:
                self.send_error(HTTPStatus.NOT_FOUND, f"no camera {cam!r}; have {sorted(studio.frames)}")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while True:
                    jpg = studio.jpeg(cam)
                    if jpg is not None:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            actions = {
                "connect": lambda: studio.connect(cameras_only=bool(body.get("cameras_only"))),
                "disconnect": studio.disconnect,
                "start": lambda: studio.start(body.get("task")),
                "save": studio.save,
                "discard": studio.discard,
                "auto": lambda: studio.set_auto(body.get("on")),
                "upload": studio.upload,
            }
            fn = actions.get(urlsplit(self.path).path.removeprefix("/api/"))
            if fn is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            fn()
            self._json(studio.status())

    return Handler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("single", "dual"), required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--task", default="")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--episode-time-s", type=float, default=25)
    p.add_argument("--reset-time-s", type=float, default=10)
    p.add_argument("--private", default="true")
    p.add_argument("--tags", default="so101,stacking,hfteama")
    p.add_argument("--views", default="", help="fixed camera slots, comma separated")
    p.add_argument("--configured", default=None, help="name:arms for every camera in CAMERA_TABLE")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args, hw_flags = p.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")

    hw = draccus.parse(Hardware, args=[f for f in hw_flags if f != "--"])
    studio = Studio(hw, args.mode, args.repo_id, args.task, args.fps, args.episode_time_s,
                    args.reset_time_s, args.private.lower() == "true", args.tags.split(","),
                    views=[v for v in args.views.split(",") if v] or None,
                    configured=None if args.configured is None else
                    dict(e.split(":", 1) for e in args.configured.split(",") if e))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(studio))
    server.daemon_threads = True
    print(f"Recording studio ({args.mode} -> {args.repo_id}): http://localhost:{args.port}", flush=True)
    studio.connect(cameras_only=True)  # show the cameras right away; arms connect from the page
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio._shutting_down = True
        studio.disconnect()


if __name__ == "__main__":
    main()
