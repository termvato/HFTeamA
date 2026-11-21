"""Live view of a pair's cameras in Rerun, opened exactly as recording opens them."""

from __future__ import annotations

import time

from hfteama.hardware.pairs import PairConfig


def view(cfg: PairConfig, names: list[str]) -> int:
    import rerun as rr
    from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

    selected = names or list(cfg.cameras)
    unknown = [n for n in selected if n not in cfg.cameras]
    if unknown or not selected:
        available = ", ".join(cfg.cameras) or "none"
        raise SystemExit(f"Unknown camera(s) {unknown} for pair {cfg.pair!r}; have: {available}")

    cams = {}
    for name in selected:
        params = {k: v for k, v in cfg.cameras[name].items() if k != "type"}
        print(f"Opening {name}: {params['index_or_path']}")
        cams[name] = OpenCVCamera(OpenCVCameraConfig(**params))
        cams[name].connect()

    rr.init(f"hfteama_{cfg.pair}_cameras", spawn=True)
    print("Streaming to Rerun. Ctrl+C to stop.")
    try:
        while True:
            for name, cam in cams.items():
                frame = cam.read()
                rr.log(f"cameras/{name}", rr.Image(frame).compress(jpeg_quality=80))
                rr.log(f"brightness/{name}", rr.Scalars(float(frame.mean())))
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cams.values():
            cam.disconnect()
    return 0
