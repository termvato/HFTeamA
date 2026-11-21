"""What the planner sees at one moment: tasks, arms, time, and what is already done or running.

The executor builds a Problem on every re-plan. Running steps are fixed (never moved); arms that
hold a part (a `same_arm_as` partner finished on them) must place it before anything else.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from hfteama.scheduler.plan import Assignment
from hfteama.taskgraph import Task, TaskGraph


class PlanError(RuntimeError):
    """The planner could not produce a valid plan."""


@dataclass
class Problem:
    graph: TaskGraph
    arms: list[str]
    now: float = 0.0
    done: frozenset[str] = frozenset()
    running: dict[str, Assignment] = field(default_factory=dict)  # task id -> fixed assignment
    held: dict[str, str] = field(default_factory=dict)  # arm -> task it must do next (holds part)
    restrict: dict[str, frozenset[str]] = field(default_factory=dict)  # arm -> only these tasks
    capabilities: dict[str, frozenset[str]] | None = None
    estimate: Callable[[Task, str], float] | None = None  # duration model; default task's own
    dropped: frozenset[str] = frozenset()  # failed for good: never planned again

    def task(self, tid: str) -> Task:
        return self.graph.tasks[tid]

    def duration(self, tid: str, arm: str) -> float:
        t = self.task(tid)
        return self.estimate(t, arm) if self.estimate else t.duration_on(arm)

    def pending(self) -> list[str]:
        """Tasks not done and not running, in dependency order."""
        return [t for t in self.graph.topo_order() if t not in self.done and t not in self.running]

    def capable_arms(self, tid: str) -> list[str]:
        t = self.task(tid)
        out = []
        for a in self.arms:
            caps = self.capabilities.get(a) if self.capabilities else None
            if not t.can_run_on(a, caps):
                continue
            if a in self.restrict and tid not in self.restrict[a]:
                continue
            out.append(a)
        return out

    def follower_of(self, tid: str) -> str | None:
        """The task that must run on the same arm right after `tid` (it holds the part), if any."""
        for t in self.graph.tasks.values():
            if t.same_arm_as == tid:
                return t.id
        return None

    def pinned_arm(self, tid: str) -> str | None:
        """Arm a task is tied to because its same_arm_as partner is running or held there."""
        for arm, f in self.held.items():
            if f == tid:
                return arm
        partner = self.task(tid).same_arm_as
        if partner and partner in self.running:
            return self.running[partner].arm_id
        return None


def blocked_tasks(p: Problem) -> set[str]:
    """Pending tasks no available arm can do, plus everything that depends on them."""
    blocked: set[str] = set()
    for tid in p.pending():
        t = p.task(tid)
        pin = p.pinned_arm(tid)
        arms = p.capable_arms(tid)
        if pin is not None:
            arms = [a for a in arms if a == pin]
        if tid in p.dropped or not arms or any(d in blocked for d in t.deps):
            blocked.add(tid)
        elif t.same_arm_as and t.same_arm_as in blocked:
            blocked.add(tid)
    return blocked


def unblocked_by(p: Problem, new_arm: str, capabilities: frozenset[str] | None = None) -> set[str]:
    """Steps that become possible only because `new_arm` joins (shown next to the limit)."""
    before = blocked_tasks(p)
    caps = dict(p.capabilities) if p.capabilities else None
    if caps is not None and capabilities is not None:
        caps[new_arm] = capabilities
    after = blocked_tasks(replace(p, arms=[*p.arms, new_arm], capabilities=caps))
    return before - after


def check_chains(p: Problem) -> None:
    """A same_arm_as follower may only depend on its partner and on the partner's ancestors."""
    ancestors: dict[str, set[str]] = {}
    for tid in p.graph.topo_order():
        anc: set[str] = set()
        for d in p.task(tid).deps:
            anc |= {d} | ancestors[d]
        ancestors[tid] = anc
    for t in p.graph.tasks.values():
        if t.same_arm_as is None:
            continue
        extra = set(t.deps) - {t.same_arm_as} - ancestors[t.same_arm_as]
        if extra:
            raise PlanError(f"{t.id}: same_arm_as follower also depends on {sorted(extra)}")
