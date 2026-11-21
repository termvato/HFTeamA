"""The overhead camera: found by name, with focus, exposure and white balance pinned.

Autofocus would change the focal length and invalidate the intrinsics; auto exposure and white
balance would shift the colours the piece detector compares against the table. The controls
are set with v4l2-ctl (Linux).
"""

from __future__ import annotations

import subprocess


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


class OverheadCamera:
    def __init__(self, cam_cfg: dict, device: str | None = None):
        self.cfg = cam_cfg
        self.device = device or cam_cfg.get("device") or find_device(cam_cfg.get("name", ""))
        if self.device is None:
            raise SystemExit(
                f"Overhead camera {cam_cfg.get('name')!r} not found (v4l2-ctl --list-devices)."
            )
        self.cap = None

    def open(self) -> OverheadCamera:
        import cv2

        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {self.device} (busy? fuser -v {self.device})")
        if self.cfg.get("fourcc"):  # the pixel format must be set before the size
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["height"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        got = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if got != (self.cfg["width"], self.cfg["height"]):
            cap.release()
            raise SystemExit(f"Camera gave {got}, the intrinsics are for the configured size.")
        failed = apply_controls(self.device, self.cfg.get("controls", {}))
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
