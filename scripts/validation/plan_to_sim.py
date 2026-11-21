"""VLM plan format -> reach -> executor -> Action's SkillWorker and IK in MuJoCo, real layout.

Needs Action's arm code (branch action-arm-skills) and MuJoCo. No hardware.
Run from the repo root: python scripts/validation/plan_to_sim.py
"""

import math
import tempfile
import time
from pathlib import Path

from hfteama.arms.poses import Poses
from hfteama.arms.sim import SimTable
from hfteama.arms.worker import SkillWorker

from hfteama.arms import generic_skills  # noqa
from hfteama.events import EventLog
from hfteama.executor import Executor
from hfteama.perception.decomposer import load_bins, sort_steps
from hfteama.perception.scene import SceneObject
from hfteama.registry import load_catalog, resolve
from hfteama.taskgraph import TaskGraph

cfg = "configs/hacklab_table.json"
cat = load_catalog(cfg)
bowl = load_bins(cfg)[0]
# five objects where a VLM scan could put them (table frame, z = half the assumed 4 cm height)
pts = {
    "obj_01": (0.10, 0.10),
    "obj_02": (0.22, 0.02),
    "obj_03": (0.45, 0.05),
    "obj_04": (0.55, 0.15),
    "obj_05": (0.30, 0.30),
}
print("SceneObject fields:", [f for f in SceneObject.__dataclass_fields__])
tasks = {}
for oid, (x, y) in pts.items():
    obj = SceneObject(id=oid, model="cube", label="cube", xyz=(x, y, 0.02), quat_wxyz=(1, 0, 0, 0))
    for t in sort_steps(obj, "things", bowl):
        tasks[t.id] = t
res = resolve(TaskGraph("vlm", tasks), cat.values(), travel_s_per_m=10.0)
g = res.graph
for tid, t in g.tasks.items():
    if t.skill == "pick":
        print(f"  {tid}: allowed {t.allowed_arms}")
print("blocked by reach:", getattr(res, "blocked", None))
poses = Poses(
    objects={o: [x, y, 0.02] for o, (x, y) in pts.items()},
    arms={
        a: {"base": list(n.base_pose_m), "yaw": math.radians(n.base_yaw_deg)}
        for a, n in cat.items()
    },
)
log = EventLog(Path(tempfile.mkdtemp()) / "e.jsonl")
with SimTable(poses, arm_ids=list(cat), speed=8.0) as table:
    ex = Executor(g, log, max_retries=1)
    for a in cat:
        ex.add_arm(SkillWorker(a, table.driver(a), poses, backend="sim", clock=table.sim_time))
    t0 = time.time()
    ex.start()
    done = ex.wait(180)
    ex.stop()
    print("finished:", done, "wall", round(time.time() - t0, 1), "s")
    st = ex.snapshot()["status"]
    print("status:", {s: sum(v == s for v in st.values()) for s in set(st.values())})
for e in log.read():
    if e.kind == "task_end":
        print(
            f"  {e.task:14s} {e.arm or '-':16s} {e.outcome:8s} {e.data.get('duration', 0):5.1f}s {e.data.get('reason', '')}"
        )
