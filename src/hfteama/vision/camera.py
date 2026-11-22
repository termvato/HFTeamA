"""The overhead camera: found by name, with focus, exposure and white balance pinned.

Autofocus would change the focal length and invalidate the intrinsics; auto exposure and white
balance would shift the colours the piece detector compares against the table. The controls
are set with v4l2-ctl (Linux). On Windows the camera is found by its DirectShow name (through
ffmpeg's device list, same order as OpenCV's CAP_DSHOW indices) and the same controls are mapped
onto OpenCV properties; `"device"` in the config (a DirectShow index) overrides the lookup.
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
import threading


def find_device(name: str) -> str | None:
    """/dev/videoN of the first capture node of the camera whose name contains `name`."""
    try:
        out = subprocess.run(
            ["v4l2-ctl", "--list-devices"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        return None
    inside = False
    for line in out.splitlines():
        if line and not line.startswith(("\t", " ")):
            inside = name in line
        elif inside and "/dev/video" in line:
            return line.strip()
    return None


def apply_controls(device: str, controls: dict) -> list[str]:
    """Set each control in order (auto modes first, then values); return the ones that failed."""
    failed = []
    for k, v in controls.items():
        r = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={k}={v}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode:
            failed.append(k)
    return failed


WINDOWS = sys.platform == "win32"


def dshow_devices() -> list[str]:
    """Windows: DirectShow video device names, in the order OpenCV's CAP_DSHOW indexes them."""
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return []
    out = subprocess.run(
        [ff, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
        check=False,
    ).stderr
    return re.findall(r'"([^"]+)" \(video\)', out)


def find_device_windows(name: str) -> str | None:
    """Windows: the DirectShow index (as a string) of the first camera whose name has `name`."""
    for i, dev in enumerate(dshow_devices()):
        if name.lower() in dev.lower():
            return str(i)
    return None


def apply_controls_windows(cap, controls: dict) -> list[str]:
    """The v4l2 controls of the config, mapped onto OpenCV's DirectShow properties."""
    import cv2

    mapping = {  # auto modes first, then values (as on Linux)
        "focus_automatic_continuous": cv2.CAP_PROP_AUTOFOCUS,
        "white_balance_automatic": cv2.CAP_PROP_AUTO_WB,
        "focus_absolute": cv2.CAP_PROP_FOCUS,
        "white_balance_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
        "exposure_time_absolute": cv2.CAP_PROP_EXPOSURE,
        "gain": cv2.CAP_PROP_GAIN,
        "backlight_compensation": cv2.CAP_PROP_BACKLIGHT,
        "zoom_absolute": cv2.CAP_PROP_ZOOM,
    }
    failed = []
    for key, prop in mapping.items():
        if key not in controls:
            continue
        value = float(controls[key])
        if key == "exposure_time_absolute":  # v4l2: units of 100 us; DirectShow: log2(seconds)
            value = float(round(math.log2(max(value * 1e-4, 1e-4))))
        if not cap.set(prop, value):
            failed.append(key)
    return failed


class FfmpegCapture:
    """Windows fallback with cv2.VideoCapture's grab/read/release: one ffmpeg process streams
    raw BGR frames; a thread keeps only the newest, so read() never returns a stale frame."""

    def __init__(self, name: str, cfg: dict):
        import imageio_ffmpeg
        import numpy as np

        self.w, self.h = int(cfg["width"]), int(cfg["height"])
        self._np = np
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
               "-f", "dshow", "-rtbufsize", "64M", "-video_size", f"{self.w}x{self.h}"]  # fmt: skip
        if cfg.get("fourcc", "").upper() == "MJPG":
            cmd += ["-vcodec", "mjpeg"]
        cmd += ["-i", f"video={name}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._latest = None
        self._seq = 0
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._pump, name="ffmpeg-camera", daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        size = self.w * self.h * 3
        out = self.proc.stdout
        while True:
            buf = out.read(size) if out else b""
            if len(buf) < size:
                break
            img = self._np.frombuffer(buf, self._np.uint8).reshape(self.h, self.w, 3)
            with self._cond:
                self._latest, self._seq = img, self._seq + 1
                self._cond.notify_all()
        with self._cond:
            self._latest = None
            self._seq = -1
            self._cond.notify_all()

    def isOpened(self) -> bool:  # noqa: N802 (cv2's name)
        return self.proc.poll() is None

    def grab(self) -> bool:
        return self.isOpened()

    def read(self, timeout: float = 10.0):
        """The next frame after this call (so never one taken before it)."""
        with self._cond:
            seq = self._seq
            self._cond.wait_for(lambda: self._seq != seq, timeout)
            if self._seq < 0 or self._latest is None or self._seq == seq:
                return False, None
            return True, self._latest.copy()

    def release(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(5)


class OverheadCamera:
    def __init__(self, cam_cfg: dict, device: str | None = None):
        self.cfg = cam_cfg
        find = find_device_windows if WINDOWS else find_device
        self.device = device or cam_cfg.get("device") or find(cam_cfg.get("name", ""))
        if self.device is None:
            where = "DirectShow devices" if WINDOWS else "v4l2-ctl --list-devices"
            raise SystemExit(f"Overhead camera {cam_cfg.get('name')!r} not found ({where}).")
        self.cap = None

    def open(self) -> OverheadCamera:
        import cv2

        if WINDOWS:
            cap = cv2.VideoCapture(int(self.device), cv2.CAP_DSHOW)
            if not cap.isOpened():  # some setups only let ffmpeg open the camera
                cap = FfmpegCapture(dshow_devices()[int(self.device)], self.cfg)
                print(
                    "[camera] reading frames through ffmpeg; focus, exposure and white"
                    " balance are not pinned (camera defaults)"
                )
                self.cap = cap
                for _ in range(10):
                    cap.read()
                return self
        else:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open camera {self.device} (in use by another program?)")
        if self.cfg.get("fourcc"):  # the pixel format must be set before the size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != (self.cfg["width"], self.cfg["height"]):
            cap.release()
            raise SystemExit(f"Camera gave {got}, the intrinsics are for the configured size.")
        controls = self.cfg.get("controls", {})
        if WINDOWS:
            failed = apply_controls_windows(cap, controls)
        else:
            failed = apply_controls(self.device, controls)
        if failed:
            print(f"[camera] could not set: {', '.join(failed)}")
        self.cap = cap
        for _ in range(10):  # let exposure and focus settle
            cap.read()
        return self

    def frame(self):
        """A fresh frame (the driver may hold a few old ones: drop them)."""
        for _ in range(3):
            self.cap.grab()
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"No frame from {self.device}")
        return img

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
