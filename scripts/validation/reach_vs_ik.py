"""Does the planner's reach agree with the IK? Real table layout (configs/hacklab_table.json).

Needs Action's arm code (arms/ik.py, arms/generic_skills.py: branch action-arm-skills) and MuJoCo.
No hardware. Run from the repo root: python scripts/validation/reach_vs_ik.py
"""

import math
import time

import numpy as np
from hfteama.arms.generic_skills import _pick_waypoints, _place_waypoints
from hfteama.arms.ik import ArmBase, Kinematics

from hfteama.perception.decomposer import load_bins
from hfteama.registry import load_catalog

cfg = "configs/hacklab_table.json"
cat = load_catalog(cfg)
bins = load_bins(cfg)
bowl = bins[0]
kin = {a: Kinematics(ArmBase(n.base_pose_m, math.radians(n.base_yaw_deg))) for a, n in cat.items()}
Z = 0.02  # what the VLM pointing writes: half of object_height_m = 0.04


def ik_ok(arm, pts):
    errs = [kin[arm].solve(p, down=True) for p in pts]
    return all(r.ok for r in errs), max(r.error_m for r in errs)


# 1) the place: can each arm drop into the bowl where the config says?
print("PLACE at bowl", bowl.xyz)
for a, n in cat.items():
    ok, err = ik_ok(a, _place_waypoints(np.array(bowl.xyz)))
    d = math.dist(n.pan_axis_xy, bowl.xyz[:2])
    print(
        f"  {a}: planner reach={n.reaches(bowl.xyz)} (dist {d:.3f} m)  IK ok={ok} (worst error {err * 1000:.1f} mm)"
    )

# 2) picks over the table: planner's 2D reach vs IK through every pick waypoint
xs = np.arange(-0.10, 0.70 + 1e-9, 0.04)
ys = np.arange(-0.05, 0.45 + 1e-9, 0.04)
stats = {a: dict(both=0, reach_only=0, ik_only=0, neither=0, bad=[]) for a in cat}
t0 = time.time()
for x in xs:
    for y in ys:
        p = np.array([x, y, Z])
        for a, n in cat.items():
            r = n.reaches(p)
            ok, err = ik_ok(a, _pick_waypoints(p))
            k = "both" if r and ok else "reach_only" if r else "ik_only" if ok else "neither"
            stats[a][k] += 1
            if k == "reach_only":
                stats[a]["bad"].append(
                    (
                        round(float(x), 2),
                        round(float(y), 2),
                        round(math.dist(n.pan_axis_xy, (x, y)), 3),
                    )
                )
print(f"PICK grid {len(xs)}x{len(ys)} points at z={Z} ({time.time() - t0:.0f} s)")
for a, s in stats.items():
    bad = s.pop("bad")
    print(f"  {a}: {s}")
    if bad:
        ds = sorted(b[2] for b in bad)
        print(
            f"    planner says reachable but IK fails at {len(bad)} points; distance from the pan axis: min {ds[0]} m, median {ds[len(ds) // 2]} m, max {ds[-1]} m"
        )
# radial profile: where does IK really stop?
print("IK reach by distance straight ahead of rakesh_follower (pick waypoints, tool down):")
n = cat["rakesh_follower"]
ax, ay = n.pan_axis_xy
for d in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.33, 0.36, 0.38, 0.40, 0.42, 0.44):
    p = np.array([ax, ay - d, Z])
    ok, err = ik_ok("rakesh_follower", _pick_waypoints(p))
    print(f"    {d:.2f} m: {'ok ' if ok else 'FAIL'} worst {err * 1000:5.1f} mm", end="")
    r = kin["rakesh_follower"].solve(p, down=True)
    print(f" | grasp point only: {'ok' if r.ok else 'FAIL'}")
