"""Frames from the top camera: one still through ffmpeg (DirectShow on Windows, V4L2 on Linux),
or from a file / MJPEG stream URL. OpenCV could not open the C920 by index on the Windows laptop;
ffmpeg by device name works. Needs the `camera` extra (imageio-ffmpeg)."""

from __future__ import annotations

import io
import subprocess
import sys
import urllib.request
from collections.abc import Callable

from PIL import Image


def grab(device: str, size: tuple[int, int] = (1920, 1080), settle_s: float = 2.0) -> Image.Image:
    """One frame from a camera by name ("HD Pro Webcam C920") or /dev/video path. `settle_s`
    skips the first frames, while exposure and white balance settle."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    src = ["-f", "dshow", "-i", f"video={device}"] if sys.platform == "win32" else [
        "-f", "v4l2", "-i", device]  # fmt: skip
    cmd = [ff, "-hide_banner", "-loglevel", "error", "-video_size", f"{size[0]}x{size[1]}", *src,
           "-ss", str(settle_s), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png",
           "-"]  # fmt: skip
    out = subprocess.run(cmd, capture_output=True, timeout=30, check=True).stdout
    return Image.open(io.BytesIO(out)).convert("RGB")


def load(src: str) -> Image.Image:
    """A frame from a file, or the first JPEG of an http(s) MJPEG stream / snapshot URL."""
    if not src.startswith(("http://", "https://")):
        return Image.open(src).convert("RGB")
    with urllib.request.urlopen(src, timeout=10) as r:
        buf = r.read(5_000_000)
    s = buf.find(b"\xff\xd8")
    e = buf.find(b"\xff\xd9", s + 2)
    return Image.open(io.BytesIO(buf[s : e + 2])).convert("RGB")


def frame_source(
    device: str | None = None, image: str | None = None, size: tuple[int, int] = (1920, 1080)
) -> Callable[[], Image.Image]:
    """A no-argument grab function: `image` (file or URL) wins over `device`."""
    if image:
        return lambda: load(image)
    if device:
        return lambda: grab(device, size)
    raise ValueError("give a camera device or an image")
