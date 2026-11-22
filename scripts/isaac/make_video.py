"""Turn rendered frames and the run's timeline into an MP4 with captions.

    python scripts/isaac/make_video.py runs/pyramid            # -> runs/pyramid/pyramid.mp4

Reads <run>/frames/frames.csv (scripts/isaac/mirror.py --replay --record) and <run>/timeline.json
(scripts/isaac/pyramid_demo.py --out). Draws:
  - a status panel: elapsed time, which arms are working, boxes placed;
  - a banner for each caption ("Second arm comes into action", ...), held for a few seconds;
  - a title at the start and the result at the end.
Needs Pillow and an ffmpeg binary (PATH, $FFMPEG, or the imageio-ffmpeg package).
"""

from __future__ import annotations

import argparse
import bisect
import csv
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARM_COLOURS = [(22, 22, 25), (31, 153, 71), (236, 236, 232)]  # arm 1 black, 2 green, 3 white
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BANNER_S, FADE_S = 3.5, 0.4


def find_ffmpeg() -> str:
    for c in (os.environ.get("FFMPEG"), shutil.which("ffmpeg")):
        if c:
            return c
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    hits = glob.glob(
        os.path.expanduser(
            "~/miniconda3/envs/*/lib/python3*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*"
        )
    )
    if hits:
        return hits[0]
    raise SystemExit("no ffmpeg: install it, or set FFMPEG=<path>")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_BOLD if bold else FONT, size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run")
    ap.add_argument("--out", default="")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    run = Path(a.run)
    frames = list(csv.DictReader(open(run / "frames" / "frames.csv")))
    tl = json.loads((run / "timeline.json").read_text())
    samples, captions = tl["samples"], tl["captions"]
    sample_t = [s["t"] for s in samples]
    start = captions[0]["t"]
    done = next((c for c in captions if c["text"].startswith("Pyramid")), None)
    joins = {c["arm"]: c["t"] for c in captions if c["arm"]}
    arms = sorted(joins)
    out = Path(a.out) if a.out else run / f"{run.name}.mp4"

    first = Image.open(run / "frames" / frames[0]["file"])
    w, h = first.size
    ff = subprocess.Popen(
        [
            find_ffmpeg(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(a.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    s = h / 720  # scale the layout with the frame
    f_small, f_mid, f_big = font(int(22 * s)), font(int(26 * s), True), font(int(46 * s), True)

    for row in frames:
        t = float(row["sim_t"])
        img = Image.open(run / "frames" / row["file"]).convert("RGB")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        k = max(0, bisect.bisect_right(sample_t, t) - 1)
        smp = samples[k]
        elapsed = max(0.0, min(t, done["t"] if done else t) - start)

        # status panel, top left
        x0, y0, pw, ph = int(24 * s), int(24 * s), int(330 * s), int(170 * s)
        d.rounded_rectangle((x0, y0, x0 + pw, y0 + ph), radius=int(14 * s), fill=(15, 18, 24, 170))
        d.text(
            (x0 + 18 * s, y0 + 14 * s), f"Time  {elapsed:5.1f} s", font=f_mid, fill=(255, 255, 255)
        )
        d.text((x0 + 18 * s, y0 + 56 * s), "Arms working", font=f_small, fill=(200, 205, 215))
        for i, arm in enumerate(arms):
            cx, cy, r = x0 + (190 + i * 40) * s, y0 + 70 * s, 13 * s
            on = t >= joins[arm]
            col = ARM_COLOURS[i % 3] if on else (90, 92, 98)
            d.ellipse(
                (cx - r, cy - r, cx + r, cy + r),
                fill=col + (255,),
                outline=(255, 255, 255, 200 if on else 60),
                width=2,
            )
            d.text(
                (cx, cy),
                str(i + 1),
                font=font(int(15 * s), True),
                fill=(20, 20, 22) if on and sum(col) > 500 else (255, 255, 255),
                anchor="mm",
            )
        placed, total = smp["placed"], smp["total"]
        d.text(
            (x0 + 18 * s, y0 + 98 * s),
            f"Boxes placed  {placed} / {total}",
            font=f_small,
            fill=(200, 205, 215),
        )
        bx, by, bw = x0 + 18 * s, y0 + 134 * s, pw - 36 * s
        d.rounded_rectangle(
            (bx, by, bx + bw, by + 12 * s), radius=int(6 * s), fill=(70, 74, 82, 255)
        )
        if placed:
            d.rounded_rectangle(
                (bx, by, bx + bw * placed / total, by + 12 * s),
                radius=int(6 * s),
                fill=(242, 173, 20, 255),
            )

        # banner for the latest caption, centred near the top
        live = [
            c for c in captions if c["t"] <= t < c["t"] + BANNER_S or (c is done and t >= c["t"])
        ]
        if t - start < BANNER_S:
            live = [{"t": start, "text": "3 SO-101 arms build a pyramid", "arm": "", "title": True}]
        for c in live[-1:]:
            age = t - c["t"]
            alpha = (
                1.0 if c is done else min(1.0, age / FADE_S, max(0.0, (BANNER_S - age) / FADE_S))
            )
            text = c["text"]
            if c is done:
                text = f"Pyramid complete: {total} boxes in {done['t'] - start:.1f} s"
            tw = d.textlength(text, font=f_big)
            bx0 = (w - tw) / 2 - 36 * s
            by0 = h - 150 * s
            acc = ARM_COLOURS[arms.index(c["arm"]) % 3] if c.get("arm") in arms else (255, 255, 255)
            d.rounded_rectangle(
                (bx0, by0, bx0 + tw + 72 * s, by0 + 82 * s),
                radius=int(16 * s),
                fill=(15, 18, 24, int(190 * alpha)),
            )
            d.rectangle(
                (bx0, by0 + 12 * s, bx0 + 8 * s, by0 + 70 * s),
                fill=acc + (int(255 * alpha),),
                outline=(255, 255, 255, int(200 * alpha)),
            )
            d.text(
                (w / 2, by0 + 41 * s),
                text,
                font=f_big,
                fill=(255, 255, 255, int(255 * alpha)),
                anchor="mm",
            )
            if c.get("title"):
                sub = "Simulated in MuJoCo, rendered in Isaac Sim, run by the team's Executor"
                d.text(
                    (w / 2, by0 - 22 * s),
                    sub,
                    font=f_small,
                    fill=(255, 255, 255, int(235 * alpha)),
                    anchor="mm",
                )
        frame = Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close()
    if ff.wait():
        raise SystemExit("ffmpeg failed")
    print(f"{len(frames)} frames, {len(frames) / a.fps:.1f} s -> {out}")


if __name__ == "__main__":
    main()
