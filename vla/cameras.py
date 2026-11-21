"""List the cameras, and check the configured ones together before recording.

    ./vla/cameras.sh list            # every video device: stable path, USB port, MJPG sizes, config name
    ./vla/cameras.sh check single    # open the single-arm cameras together, as recording does
    ./vla/cameras.sh check dual

`check` fails (exit 1) when a camera:
- does not open, or does not deliver the configured size, format (MJPG) or frame rate;
- sends frozen frames;
- has the same device as another camera.
It warns when cameras share one USB port or hub. It saves one labelled image of every view
(vla/camera_check.jpg) to check framing: the same view in every session.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import draccus
import numpy as np
from lerobot.cameras import CameraConfig, make_cameras_from_configs
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (registers "opencv")

SHEET = Path(__file__).with_name("camera_check.jpg")


# ---------------------------------------------------------------- devices (sysfs, no camera opened)


def by_id_paths() -> dict[str, str]:
    """/dev/videoN -> /dev/v4l/by-id/... (the stable name that survives replugging)."""
    d = Path("/dev/v4l/by-id")
    return {str(p.resolve()): str(p) for p in d.iterdir()} if d.exists() else {}


def usb_port(video: str) -> str:
    """USB bus and port of /dev/videoN, e.g. '3-1' (bus 3, port 1), or '' if not USB."""
    path = Path(f"/sys/class/video4linux/{video}/device").resolve()
    ports = [p.name for p in (path, *path.parents) if re.fullmatch(r"\d+-[\d.]+", p.name)]
    return ports[-1] if ports else ""


def mjpeg_sizes(dev: str) -> list[str]:
    """Frame sizes the camera offers as MJPG (asks ffmpeg; empty if it cannot tell)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "v4l2", "-list_formats", "all", "-i", dev],
                             capture_output=True, text=True, timeout=10).stderr
    except (OSError, subprocess.TimeoutExpired):
        return []
    line = next((ln for ln in out.splitlines() if "mjpeg" in ln.lower()), "")
    return line.split(":")[-1].split() if line else []


def users_of(dev: str) -> list[str]:
    """'pid command' of every process that has the device open (only the ones we may look at)."""
    real, found = os.path.realpath(dev), []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if any(os.path.realpath(fd) == real for fd in (proc / "fd").iterdir()):
                cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                found.append(f"{proc.name} {cmdline[:60]}")
        except OSError:
            continue
    return found


def capture_devices() -> list[dict]:
    """Video capture nodes only: each USB camera also has a metadata node (index 1) that gives no video."""
    stable = by_id_paths()
    found = []
    for sys_dev in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(p.name[5:])):
        if (sys_dev / "index").read_text().strip() != "0":
            continue
        dev = f"/dev/{sys_dev.name}"
        found.append({"dev": dev, "stable": stable.get(dev, dev), "name": (sys_dev / "name").read_text().strip(),
                      "usb": usb_port(sys_dev.name)})
    return found


def configured(table: str) -> dict[str, tuple[str, str]]:
    """CAMERA_TABLE (from config.sh) -> {resolved device: (name, arms)}."""
    out = {}
    for line in table.splitlines():
        parts = line.split()
        if len(parts) == 4 and not parts[0].startswith("#"):
            name, arms, _rot, dev = parts
            out[os.path.realpath(dev)] = (name, arms)
    return out


def cmd_list(a: argparse.Namespace) -> None:
    conf = configured(a.table)
    devs = capture_devices()
    if not devs:
        print("no cameras found")
    want = f"{a.width}x{a.height}"
    for d in devs:
        name, arms = conf.get(os.path.realpath(d["dev"]), ("", ""))
        sizes = mjpeg_sizes(d["dev"])
        mj = "MJPG " + ("ok" if want in sizes else f"has no {want}") if sizes else "MJPG unknown"
        used = f"-> {name} ({arms})" if name else "-> not in config.sh"
        print(f"{d['dev']:12} {d['name'][:28]:28} usb {d['usb'] or '-':6} {mj:22} {used}")
        print(f"{'':12} {d['stable']}")
    present = {os.path.realpath(d["dev"]) for d in devs}
    for dev, (name, _) in conf.items():
        if dev not in present:
            print(f"MISSING      {name}: {dev} is not plugged in")


# ---------------------------------------------------------------- check (opens the cameras)


@dataclass
class Cams:
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


def parse_cameras(value: str) -> dict:
    """The --robot.cameras string -> {name: CameraConfig}, parsed exactly as LeRobot parses it."""
    return draccus.parse(Cams, args=[f"--cameras={value}"]).cameras


def fourcc_of(cam) -> str:
    code = int(cam.videocapture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))


def measure(cam, seconds: float, out: dict) -> None:
    """Frames per second actually delivered, and whether the picture ever changes."""
    n, frozen, last, img = 0, 0, None, None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        try:
            img = cam.async_read(timeout_ms=1000)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"read failed: {e}"
            return
        if last is not None and np.array_equal(img[::16, ::16], last[::16, ::16]):
            frozen += 1
        last, n = img, n + 1
    out.update(fps=n / (time.perf_counter() - t0), frozen=frozen / max(n - 1, 1), img=img)


def contact_sheet(results: dict, cols: int = 3) -> np.ndarray:
    tiles = []
    for name, r in results.items():
        img = r.get("img")
        tile = (cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (480, 360)) if img is not None
                else np.full((360, 480, 3), 40, np.uint8))
        label = f"{name}  {r.get('size', '?')}  {r.get('fps', 0):.0f} fps  {'OK' if r['ok'] else 'FAIL'}"
        cv2.rectangle(tile, (0, 0), (480, 28), (0, 0, 0), -1)
        cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 230, 120) if r["ok"] else (80, 80, 240), 2)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def cmd_check(a: argparse.Namespace) -> None:
    cfgs = parse_cameras(a.cameras)
    cams = make_cameras_from_configs(cfgs)
    results: dict[str, dict] = {}
    for name, cam in cams.items():  # open all first: problems show up only with every camera running
        r = results[name] = {"ok": False, "problems": [], "dev": str(cfgs[name].index_or_path)}
        try:
            cam.connect()
        except Exception as e:  # noqa: BLE001
            busy = [u for u in users_of(r["dev"]) if not u.startswith(f"{os.getpid()} ")]
            r["problems"].append(f"busy, already open in: {'; '.join(busy)} (stop that first)" if busy
                                 else f"does not open: {str(e).strip().splitlines()[0]}")
    threads = []
    for name, cam in cams.items():
        if cam.is_connected:
            t = threading.Thread(target=measure, args=(cam, a.seconds, results[name]))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    ports: dict[str, list[str]] = {}
    for name, cam in cams.items():
        r, cfg = results[name], cfgs[name]
        if cam.is_connected:
            got = fourcc_of(cam)
            if cfg.fourcc and got != cfg.fourcc:
                r["problems"].append(f"format {got}, not {cfg.fourcc} (uses far more USB bandwidth)")
            if "error" in r:
                r["problems"].append(r["error"])
            if r.get("img") is not None:
                h, w = r["img"].shape[:2]
                r["size"] = f"{w}x{h}"
                if (h, w) != (cam.height, cam.width):
                    r["problems"].append(f"frames are {w}x{h}, expected {cam.width}x{cam.height}")
            if r.get("fps", 0) < 0.9 * cfg.fps:
                r["problems"].append(f"{r.get('fps', 0):.1f} fps delivered, needs {cfg.fps} "
                                     "(USB bandwidth, light too low for the exposure, or a bad cable)")
            if r.get("frozen", 0) > 0.5:
                r["problems"].append("picture does not change (frozen or covered)")
            cam.disconnect()
        r["ok"] = not r["problems"]
        port = usb_port(Path(os.path.realpath(r["dev"])).name)
        if port:
            ports.setdefault(port.split(".")[0], []).append(name)

    for name, r in results.items():
        status = "OK  " if r["ok"] else "FAIL"
        print(f"{status} {name:10} {r.get('size', '-'):9} {r.get('fps', 0):5.1f} fps  {r['dev']}")
        for p in r["problems"]:
            print(f"       - {p}")
    for port, names in ports.items():
        if len(names) > 1:
            print(f"note: {', '.join(names)} share USB port/hub {port}; if fps drops, move one to another port")
    if results:
        cv2.imwrite(str(SHEET), contact_sheet(results))
        print(f"views: {SHEET}")
    sys.exit(0 if all(r["ok"] for r in results.values()) else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("--table", default="")
    s.add_argument("--width", type=int, default=640)
    s.add_argument("--height", type=int, default=480)
    s.set_defaults(fn=cmd_list)
    s = sub.add_parser("check")
    s.add_argument("--cameras", required=True, help="the --robot.cameras value lib.sh builds")
    s.add_argument("--seconds", type=float, default=5.0)
    s.set_defaults(fn=cmd_check)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
