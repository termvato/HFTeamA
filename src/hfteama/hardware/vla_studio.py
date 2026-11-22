"""Recording page for one VLA task: three cameras, joint charts, start / save / discard / delete.

    hfteama vla box_right studio          # then open http://localhost:8765

The recording engine is the Action team's `vla/studio.py` (`Studio`), used unchanged: one control
loop that drives the arms, shows the cameras and writes the same LeRobot dataset as
lerobot-record. This module only feeds it our setup and serves our own page:

- the 'bimanual' pairs as one robot, with all cameras top-level (`front`, `left_wrist`,
  `right_wrist`, the same dataset keys as `hfteama vla <task> record`);
- the task's fixed dataset and instruction from configs/vla_tasks.json (not editable on the page);
- extra actions: delete the last saved episode, check the dataset, and upload only after the
  check passes.

Disconnecting (or stopping the server) turns the followers' torque off, like lerobot-record:
bring the arms to rest first. The page says so before it disconnects.
"""

from __future__ import annotations

import sys
import threading
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from hfteama.hardware import vla_data
from hfteama.hardware.pairs import PROJECT_ROOT
from hfteama.hardware.vla_data import VlaTask

VLA_DIR = PROJECT_ROOT / "vla"
PAGE = Path(__file__).with_name("vla_studio.html")
VIEWS = ["front", "left_wrist", "right_wrist"]
IDLE_STATES = ("idle", "preview", "disconnected")


def load_engine():
    """vla/studio.py as a module (it imports its neighbours, e.g. detect_arms, by plain name)."""
    if str(VLA_DIR) not in sys.path:
        sys.path.insert(0, str(VLA_DIR))
    import studio  # noqa: PLC0415 - vla/studio.py

    return studio


def make_studio(task: VlaTask, cameras: dict | None = None, pairs_config=None, engine=None):
    """A Studio for this task: our arms and cameras, the task's dataset under dataset_root()."""
    import draccus

    engine = engine or load_engine()
    flags = vla_data.hardware_args(task, cameras, pairs_config, wrists_top_level=True)
    hw = draccus.parse(engine.Hardware, args=flags)
    views = [v for v in VIEWS if v in hw.robot.cameras] + [
        c for c in hw.robot.cameras if c not in VIEWS
    ]
    studio = engine.Studio(
        hw,
        "dual",
        task.repo_id,
        task.task,
        task.fps,
        task.episode_time_s,
        task.reset_time_s,
        task.private,
        ["so101", "bimanual", "hfteam"],
        views=views,
    )
    studio.root = vla_data.dataset_root(task)
    return studio


class Ops:
    """The page's extra actions, run while no episode is being recorded."""

    def __init__(self, studio, task: VlaTask, pairs_config=None):
        self.studio, self.task, self.pairs_config = studio, task, pairs_config
        self.busy = threading.Lock()

    def _close_dataset(self) -> str | None:
        """Finalize the open dataset so it can be edited or read. None, or why not now."""
        s = self.studio
        with s.lock:
            if s.state not in IDLE_STATES:
                return f"Not while {s.state}."
            if s.dataset is not None:
                s.dataset.finalize()
                s.dataset = None  # the next Start resumes it from disk
        return None

    def _has_data(self) -> bool:
        return (self.studio.root / "meta" / "info.json").is_file()

    def dataset(self) -> dict[str, Any]:
        """Episodes on disk with their lengths (from the metadata only)."""
        info: dict[str, Any] = {
            "task": self.task.task,
            "name": self.task.name,
            "repo_id": self.task.repo_id,
            "root": str(self.studio.root),
            "target": self.task.episodes,
        }
        if not self._has_data() or self.studio.dataset is not None:
            info["lengths_s"] = None  # being written: the page shows the live count only
            return info
        try:
            info.update(vla_data.status(self.task, root=self.studio.root))
        except Exception as e:  # noqa: BLE001 - shown on the page
            info["error"] = f"{type(e).__name__}: {e}"
        return info

    def delete_last(self) -> dict[str, Any]:
        if not self.busy.acquire(blocking=False):
            return {"ok": False, "message": "Busy."}
        try:
            why = self._close_dataset()
            if why:
                return {"ok": False, "message": why}
            if not self._has_data():
                return {"ok": False, "message": "No episodes yet."}
            n = vla_data.status(self.task, root=self.studio.root)["episodes"]
            if n == 0:
                return {"ok": False, "message": "No episodes yet."}
            kept = vla_data.discard(self.task, [n - 1], root=self.studio.root)
            left = n - 1 if self._has_data() else 0
            return {
                "ok": True,
                "message": f"Deleted episode {n - 1}. {left} left. Previous version: {kept.name}",
            }
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "message": f"Delete failed: {type(e).__name__}: {e}"}
        finally:
            self.busy.release()

    def check(self) -> dict[str, Any]:
        if not self.busy.acquire(blocking=False):
            return {"ok": False, "message": "Busy.", "problems": []}
        try:
            why = self._close_dataset()
            if why:
                return {"ok": False, "message": why, "problems": []}
            if not self._has_data():
                return {"ok": False, "message": "No episodes yet.", "problems": []}
            problems = vla_data.verify(self.task, self.studio.root, self.pairs_config)
            msg = "Check OK." if not problems else f"Check found {len(problems)} problem(s)."
            return {"ok": not problems, "message": msg, "problems": problems}
        finally:
            self.busy.release()

    def upload(self) -> dict[str, Any]:
        result = self.check()
        if not result["ok"]:
            result["message"] = "Not uploaded. " + result["message"]
            return result
        self.studio.upload()  # pushes in the background; the status shows progress and the URL
        return {"ok": True, "message": "Check OK. Uploading...", "problems": []}


def make_handler(studio, ops: Ops, engine):
    base = engine.make_handler(studio)

    class Handler(base):
        def _send_page(self) -> None:
            body = PAGE.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                self._send_page()
            elif path == "/api/dataset":
                self._json(ops.dataset())
            else:
                super().do_GET()

        def do_POST(self):  # noqa: N802
            path = urlsplit(self.path).path.removeprefix("/api/")
            extra = {"delete_last": ops.delete_last, "check": ops.check, "upload": ops.upload}
            if path not in extra:
                super().do_POST()
                return
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            self._json(extra[path]())

    return Handler


def serve(task: VlaTask, port: int = 8765, host: str = "127.0.0.1", pairs_config=None) -> int:
    import logging

    cams = vla_data.resolve_cameras(task.cameras)
    vla_data.set_camera_controls(task.cameras, cams)
    engine = load_engine()
    studio = make_studio(task, cams, pairs_config, engine)
    ops = Ops(studio, task, pairs_config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    server = ThreadingHTTPServer((host, port), make_handler(studio, ops, engine))
    server.daemon_threads = True
    print(f"VLA studio for {task.name} ({task.repo_id}): http://localhost:{port}", flush=True)
    print("Stop with ONE Ctrl-C, with the arms at rest: it turns the followers' torque off.")
    studio.connect(cameras_only=True)  # cameras right away; the arms connect from the page
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio._shutting_down = True
        studio.disconnect()
    return 0
