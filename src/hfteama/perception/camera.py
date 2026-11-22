"""Frames from the top camera: one still through ffmpeg (DirectShow on Windows, V4L2 on Linux),
or from a file / MJPEG stream URL. OpenCV could not open the C920 by index on the Windows laptop;
ffmpeg by device name works. Needs the `camera` extra (imageio-ffmpeg)."""

from __future__ import annotations

import io
import subprocess
import sys
import threading
import time
import urllib.request
from collections.abc import Callable

from PIL import Image


def dshow_name(device: str) -> str:
    """The DirectShow name as Windows lists it, matched ignoring case ("C922 Pro Stream Webcam"
    is listed as "c922 Pro Stream Webcam" on some laptops, and ffmpeg wants it exact)."""
    if sys.platform != "win32":
        return device
    import re

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-list_devices", "true",
           "-f", "dshow", "-i", "dummy"]  # fmt: skip
    try:
        err = subprocess.run(cmd, capture_output=True, timeout=10).stderr.decode(errors="replace")
    except (OSError, subprocess.TimeoutExpired):
        return device
    names = re.findall(r'"([^"]+)" \(video\)', err)
    return next((n for n in names if n.lower() == device.lower()), device)


def grab(device: str, size: tuple[int, int] = (1920, 1080), settle_s: float = 2.0) -> Image.Image:
    """One frame from a camera by name ("HD Pro Webcam C920") or /dev/video path. `settle_s`
    skips the first frames, while exposure and white balance settle."""
    import imageio_ffmpeg

    device = dshow_name(device)
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


class CameraFeed:
    """One camera kept open by a single ffmpeg process, a few frames a second.

    A camera opens for one program at a time on Windows (DirectShow), so the dashboard's live
    view and the pipeline's scans share this feed: `grab()` hands the pipeline a frame taken
    after the call (the arms may have moved), `jpeg()` gives the screen the latest one. Raw RGB
    frames, so the tag detection sees no JPEG artefacts. ffmpeg is restarted if it exits (cable
    out and back in)."""

    def __init__(self, device: str, size: tuple[int, int] = (1920, 1080), fps: float = 4.0):
        self.device, self.size, self.fps = device, tuple(size), fps
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0
        self._t = 0.0
        self._jpeg: tuple[int, int, bytes] | None = None  # (seq, max_side, bytes)
        self.error = ""
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(target=self._run, name="camera-feed", daemon=True)
        self._thread.start()

    def _cmd(self) -> list[str]:
        import imageio_ffmpeg

        w, h = self.size
        src = (
            ["-f", "dshow", "-rtbufsize", "64M", "-i", f"video={dshow_name(self.device)}"]
            if sys.platform == "win32"
            else ["-f", "v4l2", "-i", self.device]
        )
        return [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                "-video_size", f"{w}x{h}", *src, "-vf", f"fps={self.fps},scale={w}:{h}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]  # fmt: skip

    def _run(self) -> None:
        w, h = self.size
        n = w * h * 3
        while not self._stop.is_set():
            try:
                self._proc = subprocess.Popen(
                    self._cmd(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=n
                )
                out = self._proc.stdout
                assert out is not None
                while not self._stop.is_set():
                    buf = out.read(n)
                    if len(buf) < n:
                        break
                    with self._cond:
                        self._frame, self._t = buf, time.time()
                        self._seq += 1
                        self.error = ""
                        self._cond.notify_all()
                err = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
                self.error = err.strip().splitlines()[-1] if err.strip() else "camera stopped"
            except Exception as exc:  # noqa: BLE001  (keep trying: the cable may come back)
                self.error = repr(exc)
            finally:
                if self._proc is not None and self._proc.poll() is None:
                    self._proc.kill()
            self._stop.wait(2.0)

    def grab(self, timeout: float = 15.0, fresh: int = 2) -> Image.Image:
        """A frame taken after this call: `fresh` frames on, so none was buffered before it."""
        with self._cond:
            want = self._seq + fresh
            if not self._cond.wait_for(lambda: self._seq >= want, timeout):
                raise RuntimeError(f"camera {self.device!r}: no frame ({self.error or 'timeout'})")
            buf = self._frame
        assert buf is not None
        return Image.frombytes("RGB", self.size, buf)

    def jpeg(self, max_side: int = 960, quality: int = 75) -> bytes | None:
        """The latest frame as a small JPEG for the screen (None before the first frame)."""
        with self._cond:
            buf, seq = self._frame, self._seq
            cached = self._jpeg
        if buf is None:
            return None
        if cached is not None and cached[0] == seq and cached[1] == max_side:
            return cached[2]
        im = Image.frombytes("RGB", self.size, buf)
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=quality)
        data = out.getvalue()
        with self._cond:
            self._jpeg = (seq, max_side, data)
        return data

    @property
    def age(self) -> float | None:
        """Seconds since the last frame (None before the first)."""
        return None if self._t == 0.0 else time.time() - self._t

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()


_feeds: dict[str, CameraFeed] = {}
_feeds_lock = threading.Lock()


def shared_feed(device: str, size: tuple[int, int] = (1920, 1080)) -> CameraFeed:
    """The process's one feed for this camera (started on first use)."""
    with _feeds_lock:
        if device not in _feeds:
            _feeds[device] = CameraFeed(device, size)
        return _feeds[device]


def active_feed() -> CameraFeed | None:
    """The running feed, if any: what the dashboard shows."""
    with _feeds_lock:
        return next(iter(_feeds.values()), None)


def frame_source(
    device: str | None = None,
    image: str | None = None,
    size: tuple[int, int] = (1920, 1080),
    feed: bool = False,
) -> Callable[[], Image.Image]:
    """A no-argument grab function: `image` (file or URL) wins over `device`. `feed=True` keeps
    the camera open (`shared_feed`), so the dashboard can show it live between scans."""
    if image:
        return lambda: load(image)
    if device:
        if feed:
            cam = shared_feed(device, size)
            return cam.grab
        return lambda: grab(device, size)
    raise ValueError("give a camera device or an image")
