"""Record the simulated arms sorting a batch to an .mp4 (default: the real table).

    python scripts/record_sim_video.py                     # the real table (hacklab), a grasp miss
    python scripts/record_sim_video.py --layout sort        # the original 3-arm ring
    python scripts/record_sim_video.py --inject rakesh_follower:grasp --inject vato_follower:grasp

It is the same wiring as `python -m hfteama.runner --backend sim`: executor, planner, SkillWorker,
simulated arms. Frames are taken every 1/fps seconds of SIMULATED time, so the video plays at true
speed (12 fps by default: shadows off, so rendering keeps up with the sim). Needs offscreen
rendering (`MUJOCO_GL=egl` is set here) and `pip install imageio imageio-ffmpeg`.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# WSL's D3D12 GPU path crashes under a threaded renderer; Mesa software rendering is stable.
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("GALLIUM_DRIVER", "llvmpipe")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from hfteama import runner  # noqa: E402
from hfteama.arms.sim import SimTable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CAPTION = "HFTeamA | MuJoCo {ver} | {n} x SO-101 | sim t = {t:5.1f} s"


# Camera per layout: lookat (x, y, z), distance, azimuth, elevation. 'hacklab' is the real table.
LAYOUTS: dict[str, dict] = {
    "sort": {
        "task": "tasks/sort/sort12.json",
        "poses": "tasks/sort/poses.json",
        "arms": "arm_a,arm_b,arm_c",
        "inject": ["arm_b:grasp"],
        "cam": ((0.0, 0.0, 0.04), 0.95, 90.0, -52.0),
    },
    "hacklab": {
        "task": "tasks/hacklab/hacklab12.json",
        "poses": "tasks/hacklab/poses.json",
        "arms": "rakesh_follower,vato_follower",
        "inject": ["vato_follower:grasp"],
        "cam": ((0.29, 0.26, 0.04), 1.05, 90.0, -48.0),
    },
}


def _make_recorder(
    out: Path, fps: int, width: int, height: int, shadows: bool, cam_spec: tuple
) -> type[SimTable]:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    class Recording(SimTable):
        def __enter__(self) -> Recording:
            super().__enter__()
            self._model.vis.global_.offwidth = max(width, 640)
            self._model.vis.global_.offheight = max(height, 480)
            self._frames = 0
            self._rec_done = threading.Event()
            threading.Thread(target=self._record, name="recorder", daemon=True).start()
            return self

        def __exit__(self, *exc: object) -> None:
            super().__exit__(*exc)
            self._rec_done.wait(10.0)

        def _record(self) -> None:
            writer = imageio.get_writer(str(out), fps=fps, codec="libx264", macro_block_size=1)
            renderer = mujoco.Renderer(self._model, height, width)  # created in this thread (GL)
            cam = mujoco.MjvCamera()
            lookat, cam.distance, cam.azimuth, cam.elevation = cam_spec
            cam.lookat[:] = lookat
            next_t = self.sim_time()
            try:
                while not self._quit.is_set():
                    t = self.sim_time()
                    if t < next_t:
                        time.sleep(0.002)
                        continue
                    with self._lock:
                        renderer.update_scene(self._data, cam)
                    if not shadows:  # shadows triple the frame time: true speed needs them off
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                    img = Image.fromarray(renderer.render())
                    ImageDraw.Draw(img).text(
                        (12, 10),
                        CAPTION.format(ver=mujoco.__version__, n=len(self.arm_ids), t=t),
                        fill=(255, 255, 255),
                    )
                    writer.append_data(np.asarray(img))
                    self._frames += 1
                    next_t += 1.0 / fps
            finally:
                writer.close()
                renderer.close()
                self._rec_done.set()

    return Recording


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="hacklab")
    ap.add_argument("--out", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--poses", default=None)
    ap.add_argument("--arms", default=None)
    ap.add_argument("--inject", action="append", default=None, metavar="ARM:KIND")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument(
        "--shadows", action="store_true", help="prettier, but only ~5 fps: a time-lapse"
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args(argv)

    lay = LAYOUTS[args.layout]
    out = Path(args.out or ROOT / "runs" / f"sim_{args.layout}.mp4")
    task = args.task or str(ROOT / lay["task"])
    poses = args.poses or str(ROOT / lay["poses"])
    arms = args.arms or lay["arms"]
    inject = args.inject if args.inject is not None else lay["inject"]
    out.parent.mkdir(parents=True, exist_ok=True)
    runner.SimTable = _make_recorder(
        out, args.fps, args.width, args.height, args.shadows, lay["cam"]
    )  # type: ignore[misc]
    argv2 = ["--backend", "sim", "--yes", "--task", task, "--poses", poses]
    argv2 += ["--arms", arms, "--log", str(out.with_suffix(".events.jsonl"))]
    argv2 += [
        "--timeout-factor",
        "6",
    ]  # rendering steals CPU: do not cancel steps that are just late
    for spec in inject:
        argv2 += ["--inject", spec]
    rc = runner.main(argv2)
    print(f"video: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
