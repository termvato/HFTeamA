"""Write poses.json and pyramid14.json: 3 arms build a 3-layer pyramid of 14 boxes (Isaac demo).

    python tasks/pyramid/make_pyramid.py [--seed 7]

Layout (table frame, metres; x = "south" in the brief): an equilateral triangle of side 0.59 m,
every base 0.34 m from the pyramid's centre.
  arm_1 (0, 0) and arm_2 (0, 0.59) face south (+x); arm_3 (0.51, 0.29) faces north (-x).

Boxes: 14 of 40 mm (white, an AprilTag on top in Isaac Sim), scattered at random (the seed makes
it repeatable) just outside the middle circle (0.19-0.32 m from its centre). Each arm has its own
wedge of the table, the third on its side; its boxes lie inside it, 25-48 deg either side of the
arm's direction (closer in is too near its base), where the IK can pick them with the jaws
square: 5 for arm 1, 5 for arm 2, 4 for arm 3.

Two steps per box, run by the team's Executor (pick_square / place_square, in
scripts/isaac/pyramid_demo.py): the pick grips the box and lifts it straight up, still in the
arm's own wedge; the place moves it straight to its slot, puts it down and stands the arm up
over its base. Arm 1's boxes make the bottom layer first (it starts alone), then arm 2's, then
arm 3's; a box waits for the boxes under its slot.

Safety: no other arm ever enters an arm's wedge, so picks need no lock; the pyramid is the one
shared area, and a place holds its lock. The Executor only starts a step once its lock is free,
so no arm ever waits mid-motion: no timeouts, no deadlock. While one arm places, the others pick.
(Tried and dropped: locks taken by the arms while moving, which timed out, deadlocked or let two
arms touch; boxes shared between neighbouring arms, which needed those locks.)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIZE, GAP = 0.040, 0.028  # box edge, gap in the pyramid: room for the jaws; measured, a box
# boxed in on all sides overlaps the gripper by up to 5 mm at 20 mm (9 mm while moving)
PITCH = SIZE + GAP
R = 0.34  # base to pyramid centre
SIDE = R * math.sqrt(3)
CENTRE = (R / 2, SIDE / 2)
ARMS = {
    "arm_1": ((0.0, 0.0), 0.0),
    "arm_2": ((0.0, SIDE), 0.0),
    "arm_3": ((R / 2 + R, SIDE / 2), math.pi),
}
REST = [0.0, -1.7, 0.0, 0.0, 0.0]  # standing straight up
REACH_DZ = (-0.007, 0.0, 0.08, 0.10)  # grip (jaws centred on the box), box, hover, lift
RING = (0.19, 0.32)  # boxes lie this far from the pyramid's centre: near the middle circle
CLEAR = {"pyramid": 0.19, "base": 0.17, "box": 0.065}  # minimum distances (m)
N_BOXES = 14


def r4(v: float) -> float:
    return round(float(v), 4)


def angle_of(arm: str) -> float:
    (x, y), _ = ARMS[arm]
    return math.atan2(y - CENTRE[1], x - CENTRE[0])


def at(radius: float, ang: float) -> tuple[float, float]:
    return CENTRE[0] + radius * math.cos(ang), CENTRE[1] + radius * math.sin(ang)


def zone_of(x: float, y: float) -> str:
    """The arm whose base is nearest: the table is split into one zone per arm."""
    return min(ARMS, key=lambda a: math.dist((x, y), ARMS[a][0]))


WEDGE_DEG = (25, 48)  # boxes lie 25-48 deg either side of their arm's direction (around the
# pyramid): closer to its own line they are too near its base to reach; wedges are 120 deg, so
# boxes of neighbouring arms are at least 24 deg (8 cm or more) apart
PER_ARM = {"arm_1": 5, "arm_2": 5, "arm_3": 4}
MAX_PLACE_M = 0.40  # an arm places no further than this from its base (stalls near 0.43)


def slots():
    out = []  # (layer, i, j, xyz), bottom layer first
    for layer, n in enumerate((3, 2, 1)):
        for i in range(n):
            for j in range(n):
                x = CENTRE[0] + (i - (n - 1) / 2) * PITCH
                y = CENTRE[1] + (j - (n - 1) / 2) * PITCH
                z = SIZE / 2 + 0.001 + layer * SIZE
                out.append((layer, i, j, (r4(x), r4(y), r4(z))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from hfteama.arms.ik import ArmBase, Kinematics  # the team's IK decides who can reach what

    kin = {k: Kinematics(ArmBase((x, y, 0.0), yaw)) for k, ((x, y), yaw) in ARMS.items()}

    def square_ok(arm, xyz):
        """Reachable at box, hover and lift heights with the jaws square to the box."""
        for h in (0.0, math.pi / 2):
            sols = [kin[arm].solve(np.array(xyz) + [0, 0, dz], yaw=h) for dz in REACH_DZ]
            # as the skill: reached, and the jaws within 0.05 rad of square
            if all(r.ok and abs(r.heading_error_rad or 0.0) < 0.05 for r in sols):
                return True
        return False

    rng = np.random.default_rng(a.seed)
    boxes: list[tuple[tuple[float, float, float], str]] = []
    for arm, count in PER_ARM.items():  # each arm's share, at random inside its own wedge
        mine: list[tuple[float, float, float]] = []
        tries = 0
        while len(mine) < count:
            tries += 1
            assert tries < 20000, f"could not place {arm}'s boxes"
            radius = math.sqrt(rng.uniform(RING[0] ** 2, RING[1] ** 2))
            off = math.radians(rng.uniform(*WEDGE_DEG) * rng.choice([-1, 1]))
            x, y = at(radius, angle_of(arm) + off)
            if any(math.dist((x, y), b) < CLEAR["base"] for b, _ in ARMS.values()):
                continue
            if any(math.dist((x, y), b[:2]) < CLEAR["box"] for b in mine):
                continue
            xyz = (r4(x), r4(y), SIZE / 2)
            if square_ok(arm, xyz):
                mine.append(xyz)
        boxes += [(b, arm) for b in mine]
    # Which slot each arm fills. Arm 1 starts alone, so its boxes make the bottom layer first,
    # then arm 2's, then arm 3's; within a layer each arm takes the slots nearest to it (the far
    # corner can be 0.43 m away, where a stretched-out arm stalls).
    order = list(PER_ARM)
    left = {arm: [b for b, owner in boxes if owner == arm] for arm in order}
    for arm in order:  # its boxes nearest the pyramid first
        left[arm].sort(key=lambda b: math.dist(b[:2], CENTRE))
    free = list(slots())
    pairs = []  # (slot, box xyz, arm)
    for arm in order:
        for box_xyz in left[arm]:
            layer = min(sl[0] for sl in free)
            mine = [sl for sl in free if sl[0] == layer]
            sl = min(mine, key=lambda sl: math.dist(sl[3][:2], ARMS[arm][0]))
            free.remove(sl)
            pairs.append((sl, box_xyz, arm))
    pairs.sort(key=lambda p: (p[0][0], p[0][1], p[0][2]))  # bottom layer first (deps)

    objects, tasks, placed_by = {}, [], {}
    for n, ((layer, i, j, xyz), spot, arm) in enumerate(pairs, 1):
        name = f"box_{n:02d}"
        objects[name] = list(spot)
        assert all(kin[arm].solve(np.array(xyz) + [0, 0, dz]).ok for dz in (0, 0.08)), arm
        reach = math.dist(xyz[:2], ARMS[arm][0])
        assert reach <= MAX_PLACE_M, f"{arm} would place {name} {reach:.3f} m away"
        placed_by[(layer, i, j)] = name
        support = (
            [placed_by[(layer - 1, i + p, j + q)] for p in (0, 1) for q in (0, 1)] if layer else []
        )
        tasks.append(
            {
                "id": f"pick_{n:02d}",
                "skill": "pick_square",
                "args": {"object": name},
                "deps": [f"place_{s[-2:]}" for s in support],
                "allowed_arms": [arm],
                "duration": 10.0,
                "resources": [],  # in the arm's own wedge: no other arm goes there
            }
        )
        tasks.append(
            {
                "id": f"place_{n:02d}",
                "skill": "place_square",
                "args": {"object": name, "place_xyz": list(xyz)},
                "deps": [f"pick_{n:02d}", *(f"place_{s[-2:]}" for s in support)],
                "allowed_arms": [arm],
                "duration": 12.0,
                "same_arm_as": f"pick_{n:02d}",
                "resources": ["pyramid"],  # the one shared area
            }
        )
    poses = {
        "note": f"Isaac demo: 3 arms build a 14-box pyramid, boxes at random (seed {a.seed}). "
        "Written by make_pyramid.py.",
        "table_z": 0.0,
        "locations": {"pyramid": [r4(CENTRE[0]), r4(CENTRE[1]), 0.0]},
        "radii": {"pyramid": 0.12},
        "objects": objects,
        "arms": {
            arm: {"base": [r4(x), r4(y), 0.0], "yaw": r4(yaw), "rest": REST}
            for arm, ((x, y), yaw) in ARMS.items()
        },
    }
    (HERE / "poses.json").write_text(json.dumps(poses, indent=1) + "\n")
    graph = {"name": "pyramid14", "note": "Placeholder durations (s).", "tasks": tasks}
    (HERE / "pyramid14.json").write_text(json.dumps(graph, indent=1) + "\n")
    who: dict[str, int] = {}
    for t in tasks:
        if t["skill"] == "pick_square":
            key = "+".join(t["allowed_arms"])
            who[key] = who.get(key, 0) + 1
    print(f"{len(objects)} boxes (seed {a.seed}); who can take them: {who}")


if __name__ == "__main__":
    main()
