"""Three arms build a pyramid, joining one after another, run by the team's Executor (MuJoCo).

    .venv/bin/python scripts/isaac/pyramid_demo.py            # arm 2 after 3 boxes, arm 3 after 7
    .venv/bin/python scripts/isaac/pyramid_demo.py --isaac 9871   # live in Isaac Sim (mirror.py)
    .venv/bin/python scripts/isaac/pyramid_demo.py --out runs/pyramid  # also record for a video

The batch is tasks/pyramid/pyramid14.json (14 boxes, 3 layers; tasks/pyramid/make_pyramid.py).
Arm 1 starts alone. Once --join-after boxes are placed (or at --join times) the next arm is handed
to the Executor with add_arm(),
as plug-and-produce does when a real arm is plugged in: the Executor re-plans and the new arm
takes its share of the remaining steps. The other arms stay at rest until they join.

With --out DIR it writes:
  states.bin     every body pose at 30 Hz (scripts/isaac/mirror.py --replay renders it)
  timeline.json  per 0.25 s of sim time: arms working, boxes placed, predicted finish
  events.jsonl   the Executor's event log

Checks, printed at the end (exit code 1 if one fails):
  - every step ok;
  - no contact between two arms, ever (MuJoCo contacts sampled at 50 Hz), and the smallest gap
    between two arms' collision shapes over the run;
  - no arm dropped by the Executor (heartbeat lost, failed);
  - when each arm joined, and when it placed its first block.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from hfteama.arms import generic_skills, sim
from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker
from hfteama.clock import Clock
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.taskgraph import load
from hfteama.view.isaac_mirror import IsaacMirror

ROOT = Path(__file__).resolve().parent.parent.parent
TASK = ROOT / "tasks" / "pyramid" / "pyramid14.json"
ORDINAL = ["First", "Second", "Third", "Fourth"]

# ---------------------------------------------------------------- square grip (this demo only)
# Vivek's pick/place let the wrist turn freely, open the jaws fully and carry the part at the tool
# centre: fine for the sort demo, but here the jaws would pass through the boxes. pick_square and
# place_square keep the jaws square to the box from pick to place, turned for the most room; close
# them to the box's 40 mm; open them only to 52 mm (6 mm clear of each side, well inside the 28 mm
# gap to the next box); carry the box rigidly; and set it down square. Simulation only.
sim.OBJECT_HALF_M = 0.020  # 40 mm boxes
sim.GRIPPER_CLOSED = 0.35  # rad: 40.6 mm between the jaw pads
sim.GRIPPER_OPEN = 0.53  # rad: 52 mm
JAW_MID = np.array(
    [0.0002, -0.0028, 0.0068]
)  # jaw midpoint - tool centre at 0.35 rad, gripper frame
SQUARE = (0.0, np.pi / 2, np.pi, -np.pi / 2)  # jaw headings square to an axis-aligned box


def _gripper_pose(table: SimTable, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
    m, d = table._model, table._data  # noqa: SLF001
    b = m.body(f"{arm_id}/gripper").id
    return d.xpos[b].copy(), d.xquat[b].copy()


def _attach_rigid(self: SimTable, arm_id: str) -> str | None:
    obj = _attach_orig(self, arm_id)
    if obj is not None:
        with self._lock:  # noqa: SLF001
            pg, qg = _gripper_pose(self, arm_id)
            d, m = self._data, self._model  # noqa: SLF001
            pb, qb = d.xpos[m.body(obj).id].copy(), d.xquat[m.body(obj).id].copy()
            rg = np.zeros(9)
            mujoco.mju_quat2Mat(rg, qg)
            inv, rel_q = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, qg)
            mujoco.mju_mulQuat(rel_q, inv, qb)
            self._rigid = getattr(self, "_rigid", {})
            self._rigid[arm_id] = (rg.reshape(3, 3).T @ (pb - pg), rel_q)
    return obj


def _carry_rigid(self: SimTable) -> None:  # with the lock held, after every physics step
    m, d = self._model, self._data  # noqa: SLF001
    for arm_id, obj in self._attached.items():  # noqa: SLF001
        rel = getattr(self, "_rigid", {}).get(arm_id)
        if rel is None:
            continue
        pg, qg = _gripper_pose(self, arm_id)
        rg, q = np.zeros(9), np.zeros(4)
        mujoco.mju_quat2Mat(rg, qg)
        mujoco.mju_mulQuat(q, qg, rel[1])
        jid = m.body(obj).jntadr[0]
        qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        d.qpos[qadr : qadr + 3] = pg + rg.reshape(3, 3) @ rel[0]
        d.qpos[qadr + 3 : qadr + 7] = q
        d.qvel[dadr : dadr + 6] = 0.0


def _release_square(self: SimTable, arm_id: str) -> None:
    """Let go with the box upright and square (nearest quarter turn), as it settles on a flat
    face: the jaws hold it within a few degrees, enough to nudge a neighbour 6 mm away."""
    with self._lock:  # noqa: SLF001
        obj = self._attached.get(arm_id)  # noqa: SLF001
        if obj is not None:
            m, d = self._model, self._data  # noqa: SLF001
            jid = m.body(obj).jntadr[0]
            qadr = m.jnt_qposadr[jid]
            w, x, y, z = d.qpos[qadr + 3 : qadr + 7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            yaw = round(yaw / (np.pi / 2)) * (np.pi / 2)
            d.qpos[qadr + 3 : qadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    _release_orig(self, arm_id)


_release_orig = SimTable._release  # noqa: SLF001
SimTable._release = _release_square  # noqa: SLF001
_attach_orig = SimTable._attach_nearest  # noqa: SLF001
SimTable._attach_nearest = _attach_rigid  # noqa: SLF001
SimTable._carry = _carry_rigid  # noqa: SLF001


def _square(arm, points, prefer: float | None = None) -> float | None:
    """A jaw heading square to the box that the IK reaches at every point (nearest `prefer`
    first); points: one xyz or a list."""
    points = [points] if np.ndim(points) == 1 else list(points)
    options = list(SQUARE)
    if prefer is not None:
        options = sorted(SQUARE, key=lambda h: abs((h - prefer + np.pi) % (2 * np.pi) - np.pi))
    for h in options:
        sols = [arm._kin.solve(np.asarray(p, float), yaw=h) for p in points]  # noqa: SLF001
        if all(r.ok and abs(r.heading_error_rad or 0.0) < 0.05 for r in sols):
            return h
    return None


def _clearance(arm, xyz: np.ndarray, h: float, skip: str) -> float:
    """Smallest gap (m) between this arm's gripper and the boxes other than `skip`, with the tool
    centre at xyz, jaws at heading h, both open and closed (a scratch copy of the scene)."""
    t = arm._table  # noqa: SLF001
    r = arm._kin.solve(np.asarray(xyz, float), yaw=h)  # noqa: SLF001
    m = t._model  # noqa: SLF001
    scratch = mujoco.MjData(m)
    with t._lock:  # noqa: SLF001
        scratch.qpos[:] = t._data.qpos  # noqa: SLF001
    jaws = [
        g
        for g in range(m.ngeom)
        if m.body(m.geom_bodyid[g]).name
        in (f"{arm.arm_id}/gripper", f"{arm.arm_id}/moving_jaw_so101_v1")
        and m.geom_group[g] in (2, 3, 4)
    ]

    def box_xy(g: int) -> np.ndarray:
        adr = m.jnt_qposadr[m.body(m.geom_bodyid[g]).jntadr[0]]
        return scratch.qpos[adr : adr + 2]

    boxes = [  # only the boxes near enough to matter
        g
        for g in range(m.ngeom)
        if m.body(m.geom_bodyid[g]).name.startswith("box_")
        and m.body(m.geom_bodyid[g]).name != skip
        and np.linalg.norm(box_xy(g) - np.asarray(xyz, float)[:2]) < 0.12
    ]
    best, fromto = float("inf"), np.zeros(6)
    for grip in (sim.GRIPPER_OPEN, sim.GRIPPER_CLOSED):
        scratch.qpos[arm._qadr[:5]] = r.q  # noqa: SLF001
        scratch.qpos[arm._qadr[5]] = grip  # noqa: SLF001
        mujoco.mj_kinematics(m, scratch)
        for gj in jaws:
            for gb in boxes:
                best = min(best, mujoco.mj_geomDistance(m, scratch, gj, gb, 0.1, fromto))
    return best


def _roomiest(arm, points, xyz, skip: str, prefer: float | None = None) -> float | None:
    """Of the square jaw headings the IK reaches at every point, the one leaving the gripper
    furthest from the other boxes at xyz (the SO-101's gripper is lopsided: the fixed jaw carries
    the motor, so turning it half round moves that bulk to the other side)."""
    points = [points] if np.ndim(points) == 1 else list(points)
    ok = []
    for h in SQUARE:
        sols = [arm._kin.solve(np.asarray(p, float), yaw=h) for p in points]  # noqa: SLF001
        if all(r.ok and abs(r.heading_error_rad or 0.0) < 0.05 for r in sols):
            ok.append(h)
    if not ok:
        return None
    return max(ok, key=lambda h: _clearance(arm, xyz, h, skip))


def _jaw_centre_offset(arm) -> np.ndarray:
    """Where the jaws' midpoint is, relative to the tool centre the IK aims (world frame)."""
    t = arm._table  # noqa: SLF001
    with t._lock:  # noqa: SLF001
        _, qg = _gripper_pose(t, arm.arm_id)
    rg = np.zeros(9)
    mujoco.mju_quat2Mat(rg, qg)
    return rg.reshape(3, 3) @ JAW_MID


def _held_offset(arm) -> np.ndarray:
    """Box centre minus tool centre (world frame), for the box in hand."""
    t = arm._table  # noqa: SLF001
    obj = t.attached(arm.arm_id)
    with t._lock:  # noqa: SLF001
        tcp = t._tcp(arm.arm_id)  # noqa: SLF001
        pb = t._data.xpos[t._model.body(obj).id].copy()  # noqa: SLF001
    return pb - tcp


def pick_square(arm, task, poses) -> None:
    """Grip the box with the jaws square to it, then lift it straight up (still in the arm's own
    wedge). The Executor then runs this arm's place for it as soon as the pyramid is free."""
    g = generic_skills
    a = arm.arm_id
    p = g.pick_point(task, poses, a)
    box = task.args["object"]
    pick_pts = [g._above(p, dz) for dz in (g.APPROACH_DZ, 0.0, g.LIFT_DZ)]  # noqa: SLF001
    h = _roomiest(arm, pick_pts, p, box)
    if h is None:
        raise g.SkillFailed("no square grip")

    def run() -> None:
        arm.open_gripper()
        g._move(arm, g._above(p, g.APPROACH_DZ), h)  # noqa: SLF001
        grip = p - _jaw_centre_offset(arm)  # the jaws' midpoint on the box centre...
        r = arm._kin.solve(grip, yaw=h)  # noqa: SLF001
        if not (r.ok and abs(r.heading_error_rad or 0.0) < 0.05):
            grip = p  # ...or the tool centre there, if that is out of reach
        g._move(arm, grip, h)  # noqa: SLF001
        if not arm.close_gripper().holding:
            arm.open_gripper()
            g._move(arm, g._above(p, g.APPROACH_DZ), h)  # noqa: SLF001
            raise g.SkillFailed("grasp")
        g._move(arm, g._above(p, g.LIFT_DZ), h)  # noqa: SLF001

    g._retreat_on_failure(arm, p, run, h)  # noqa: SLF001


def place_square(arm, task, poses) -> None:
    """Carry the box straight to its slot, put it down square, lift out, stand up over the base.
    The Executor holds the pyramid's lock for the whole step."""
    g = generic_skills
    a = arm.arm_id
    slot = g.place_point(task, poses, a)
    box = task.args["object"]
    # any square heading keeps the box square (it turns with the gripper): take the one with the
    # most room next to the boxes already in the pyramid
    place_pts = [g._above(slot, dz) for dz in (g.APPROACH_DZ, 0.0)]  # noqa: SLF001
    hp = _roomiest(arm, place_pts, slot, box)
    if hp is None:
        raise g.SkillFailed("no square grip at the slot")

    def run() -> None:
        g._move(arm, g._above(slot, g.APPROACH_DZ), hp)  # noqa: SLF001
        g._move(arm, slot - _held_offset(arm), hp)  # noqa: SLF001  box centre on the slot
        arm.open_gripper()
        g._move(arm, g._above(slot, g.APPROACH_DZ), hp)  # noqa: SLF001  straight up first
        arm.move_joints(np.asarray(poses.arms[a]["rest"], float))  # then stand up over the base

    g._retreat_on_failure(arm, slot, run, hp)  # noqa: SLF001


generic_skills.REGISTRY["pick_square"] = pick_square
generic_skills.REGISTRY["place_square"] = place_square


class ArmSafety:
    """Contacts between different arms, and the smallest gap between two arms' collision shapes."""

    def __init__(self, table: SimTable, doing=None) -> None:
        self.table = table
        self.doing = doing  # () -> what each arm is doing, recorded at the first contact
        self.first: dict[str, tuple[float, object]] = {}
        m = table._model  # noqa: SLF001
        self.m = m
        self.arm_of = {}
        for g in range(m.ngeom):
            name = m.body(m.geom_bodyid[g]).name
            if "/" in name and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.arm_of[g] = name.split("/")[0]
        geoms = sorted(self.arm_of)
        self.pairs = [
            (a, b)
            for i, a in enumerate(geoms)
            for b in geoms[i + 1 :]
            if self.arm_of[a] != self.arm_of[b]
        ]
        # grippers vs boxes: boxes do not collide with arms in this sim, so measure instead
        self.jaws = [
            g
            for g in self.arm_of
            if m.body(m.geom_bodyid[g]).name.split("/")[-1] in ("gripper", "moving_jaw_so101_v1")
        ]
        self.boxes = {
            g: m.body(m.geom_bodyid[g]).name
            for g in range(m.ngeom)
            if m.body(m.geom_bodyid[g]).name.startswith(("box_", "block_"))
        }
        self.busy: dict[str, str] = {}  # arm -> its current step (box id), set by the sampler
        self.jaw_box_gap, self.jaw_box_what = float("inf"), ""
        self.jaw_box_at: tuple[float, object] = (0.0, None)
        self.contacts: dict[str, int] = {}
        self.min_gap, self.min_pair = float("inf"), ""
        self.min_at: tuple[float, object] = (0.0, None)
        self._fromto = np.zeros(6)
        self._n = 0

    def check(self) -> None:
        t = self.table
        with t._lock:  # noqa: SLF001
            d = t._data  # noqa: SLF001
            for c in d.contact[: d.ncon]:
                a, b = self.arm_of.get(c.geom1), self.arm_of.get(c.geom2)
                if a and b and a != b:
                    key = "-".join(sorted((a, b)))
                    self.contacts[key] = self.contacts.get(key, 0) + 1
                    if key not in self.first:
                        self.first[key] = (float(d.time), None)
            self._n += 1
            if self._n % 10 == 5:  # grippers vs boxes they are not holding, at 5 Hz
                held = set(t._attached.values())  # noqa: SLF001
                for gj in self.jaws:
                    step = self.busy.get(self.arm_of[gj]) or ""  # pick_07 / place_07 ...
                    own = f"box_{step[-2:]}" if step else None  # ... moves box_07
                    for gb, box in self.boxes.items():
                        if box in held or box == own:
                            continue
                        gap = mujoco.mj_geomDistance(self.m, d, gj, gb, 0.05, self._fromto)
                        if gap < self.jaw_box_gap:
                            self.jaw_box_gap = gap
                            self.jaw_box_at = (float(d.time), None)
                            self.jaw_box_what = (
                                f"{self.m.body(self.m.geom_bodyid[gj]).name} / {box}"
                            )
            for g1, g2 in self.pairs if self._n % 10 == 0 else ():  # gaps at 5 Hz
                gap = mujoco.mj_geomDistance(self.m, d, g1, g2, 0.2, self._fromto)
                if gap < self.min_gap:
                    self.min_gap = gap
                    self.min_at = (float(d.time), None)
                    self.min_pair = f"{self.m.body(self.m.geom_bodyid[g1]).name} / " + str(
                        self.m.body(self.m.geom_bodyid[g2]).name
                    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--task", default=str(TASK))
    ap.add_argument(
        "--join-after",
        default="3,7",
        help="boxes placed before arm 2 joins, before arm 3 joins (default 3,7)",
    )
    ap.add_argument("--join", default="", help="instead: sim seconds at which arms 2, 3 join")
    ap.add_argument("--isaac", type=int, default=0, metavar="PORT", help="stream to Isaac Sim")
    ap.add_argument("--out", default="", help="folder for states.bin, timeline.json, events.jsonl")
    ap.add_argument(
        "--hold", type=float, default=3.0, help="sim seconds to keep running at the end"
    )
    args = ap.parse_args(argv)

    task_path = Path(args.task)
    graph = load(task_path)
    poses = Poses.load(task_path.with_name("poses.json"))
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    events_path = (out or ROOT / "runs" / "pyramid") / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.unlink(missing_ok=True)
    log = EventLog(events_path)
    by_time = bool(args.join)
    joins = [float(x) for x in (args.join or args.join_after).split(",")]

    with SimTable(poses, speed=1.0) as table:
        record = out / "states.bin" if out else None
        mirror = (
            IsaacMirror(table, port=args.isaac or 9871, record=record)
            if (args.isaac or record)
            else nullcontext()
        )
        with mirror:
            arms = table.arm_ids
            ex = Executor(graph, log, clock=Clock(scale=1.0))
            timeline: list[dict] = []
            captions: list[dict] = []
            t0 = table.sim_time()
            stop = threading.Event()

            def sample() -> None:  # what the video shows, in sim time
                while not stop.is_set():
                    safety.busy = ex.snapshot()["busy"]
                    safety.check()
                    if safety.jaw_box_at[1] is None and safety.jaw_box_gap < 0.05:
                        safety.jaw_box_at = (safety.jaw_box_at[0], safety.doing())
                    if safety.min_at[1] is None and safety.min_gap < 0.2:
                        safety.min_at = (safety.min_at[0], safety.doing())
                    for key, (t_c, doing) in list(safety.first.items()):
                        if doing is None:
                            safety.first[key] = (t_c, safety.doing())
                    if timeline and table.sim_time() - timeline[-1]["t"] < 0.25:
                        stop.wait(0.02)
                        continue
                    snap = ex.snapshot()
                    status = snap["status"]
                    placed = sum(
                        1 for t, s in status.items() if t.startswith("place_") and s == "done"
                    )
                    timeline.append(
                        {
                            "t": round(table.sim_time(), 3),
                            "arms": len(snap["arms"]),
                            "placed": placed,
                            "total": sum(1 for t in status if t.startswith("place_")),
                            "finish": snap["predicted_finish"],
                            "now": snap["now"],
                        }
                    )
                    stop.wait(0.02)  # 50 Hz safety check; timeline every 0.25 s

            def join(k: int) -> None:
                arm = arms[k]
                ex.add_arm(SkillWorker(arm, table.driver(arm), poses, "sim"))
                text = (
                    f"{ORDINAL[k]} arm comes into action" if k else "First arm starts working alone"
                )
                captions.append({"t": round(table.sim_time(), 3), "text": text, "arm": arm})
                print(f"[{table.sim_time() - t0:5.1f} s] {text} ({arm})", flush=True)

            safety = ArmSafety(table)
            safety.doing = lambda: ex.snapshot()["busy"]
            join(0)
            ex.start()
            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()

            def placed() -> int:
                st = ex.snapshot()["status"]
                return sum(1 for t, v in st.items() if t.startswith("place_") and v == "done")

            for k in range(1, min(len(arms), len(joins) + 1)):
                goal = joins[k - 1]
                while not ex.finished and (
                    table.sim_time() - t0 < goal if by_time else placed() < goal
                ):
                    time.sleep(0.05)
                if ex.finished:
                    break
                join(k)
            ok = ex.wait(timeout=900)
            done_t = table.sim_time()
            captions.append({"t": round(done_t, 3), "text": "Pyramid complete", "arm": ""})
            table.wait(args.hold)
            stop.set()
            sampler.join(1.0)
            ex.stop()

    all_events = list(log.read())
    ends = [e for e in all_events if e.kind == "task_end"]
    dropped = [(e.arm, (e.data or {}).get("reason")) for e in all_events if e.kind == "arm_failed"]
    first_place = {}
    for e in ends:
        if e.outcome == "ok" and e.task.startswith("place_") and e.arm not in first_place:
            first_place[e.arm] = e.t
    t_go = next(e.t for e in all_events if e.kind == "replan")
    good = {e.task for e in ends if e.outcome == "ok"}
    by_arm: dict[str, int] = {}
    for e in ends:
        if e.outcome == "ok" and e.task.startswith("place_"):
            by_arm[e.arm] = by_arm.get(e.arm, 0) + 1
    print(
        f"{graph.name}: {len(good)}/{len(graph.tasks)} steps ok in {done_t - t0:.1f} s "
        f"(finished: {ok}); boxes placed per arm: {by_arm}"
    )
    for c in captions[:-1]:
        fp = first_place.get(c["arm"])
        first = f"first box placed at {fp - t_go:.1f} s" if fp else "placed no box"
        print(f"  {c['arm']}: joined at {c['t'] - t0:.1f} s, {first}")
    print(f"  arm-to-arm contacts: {safety.contacts or 'none'}")
    for key, (t, doing) in safety.first.items():
        print(f"    first {key} contact at {t - t0:.1f} s while: {doing}")
    print(f"  smallest gap between two arms: {safety.min_gap * 100:.1f} cm ({safety.min_pair})")
    print(f"    at {safety.min_at[0] - t0:.1f} s while: {safety.min_at[1]}")
    print(
        f"  smallest gap between a gripper and another box (not the one it moves): "
        f"{safety.jaw_box_gap * 1000:.1f} mm ({safety.jaw_box_what})"
    )
    print(f"    at {safety.jaw_box_at[0] - t0:.1f} s while: {safety.jaw_box_at[1]}")
    print(f"  arms dropped by the Executor (heartbeat / failure): {dropped or 'none'}")
    if out:
        (out / "timeline.json").write_text(
            json.dumps({"t0": t0, "captions": captions, "samples": timeline}, indent=0)
        )
        print(f"recorded to {out}")
    passed = ok and good == set(graph.tasks) and not safety.contacts and not dropped
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
