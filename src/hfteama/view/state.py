"""What the screen shows, derived only from the event log (backlog #12, #17).

Pure functions, no Dash: easy to test, and any other front end can reuse them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from hfteama.events import Event


@dataclass
class Bar:
    task: str
    arm: str
    start: float  # run seconds
    end: float
    kind: str  # "plan" | "actual" | "running"
    outcome: str = ""
    skill: str = ""


@dataclass
class Screen:
    phase: str = "empty"  # "empty" | "estimate" | "running" | "finished"
    now: float = 0.0  # run seconds
    arms: dict[str, str] = field(default_factory=dict)  # arm -> backend
    plan: list[Bar] = field(default_factory=list)
    actual: list[Bar] = field(default_factory=list)
    finish: float | None = None  # predicted (running) or actual (finished), run seconds
    estimate: float | None = None  # seconds for the whole batch with the current arms
    whatif: dict[int, float] = field(default_factory=dict)
    lower_bound: float | None = None
    binding: str = ""
    limit: str = ""
    unblocked_note: str = ""
    blocked: list[str] = field(default_factory=list)
    done: int = 0
    total: int = 0
    feed: list[str] = field(default_factory=list)  # newest first
    replanning: dict | None = None  # the re-planning moment, shown for REPLAN_HOLD_S seconds
    scene: dict = field(default_factory=dict)  # table map: arms, bins, objects (if logged)

    @property
    def remaining(self) -> float | None:
        if self.finish is None:
            return None
        return max(0.0, self.finish - self.now)


def limit_text(binding: str) -> str:
    """The binding limit in words, for the screen."""
    if binding == "arms":
        return "More arms would make this faster."
    if binding == "order":
        return "The order of the steps is the limit: more arms won't help."
    if binding.startswith("resource:"):
        name = binding.split(":", 1)[1].replace("_", " ")
        return f"The {name} is the bottleneck: more arms won't help, a second {name} would."
    return ""


def clock_text(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    s = max(0, round(seconds))
    return f"{s // 60}:{s % 60:02d}"


REPLAN_HOLD_S = 4.0  # how long the screen stays on "re-planning" after arms change
REPLAN_REASONS = {
    "arm_added": "joined",
    "arm_removed": "removed",
    "arm_failed": "lost",
    "graph_updated": "table changed",
}
PLANNER_NAMES = {"greedy": "nearest free arm", "heft": "critical work first", "cpsat": "optimal"}


def build(events: Iterable[Event], clock_now: float | None = None) -> Screen:
    """Screen state from the event log. `clock_now` is the executor clock's time (time.time()
    for real runs); without it, 'now' is the time of the last event."""
    events = list(events)
    sc = Screen()
    if not events:
        return sc
    t0: float | None = None
    plan_version: int | None = None
    plan_by_version: dict[int, list[Bar]] = {}
    starts: dict[str, Event] = {}
    last_t = events[-1].t
    for e in events:
        d = e.data
        if e.kind == "arm_added" and e.arm:
            sc.arms[e.arm] = d.get("backend", "")
            sc.feed.insert(0, f"Arm {e.arm} joined")
        elif e.kind in ("arm_removed", "arm_failed") and e.arm:
            sc.arms.pop(e.arm, None)
            if e.kind == "arm_failed":
                sc.feed.insert(0, f"Arm {e.arm} lost ({d.get('reason', '')}), moving its work")
            else:
                sc.feed.insert(0, f"Arm {e.arm} removed: going home, then torque off")
        elif e.kind == "arm_parked" and e.arm:
            if d.get("ok", True):
                sc.feed.insert(0, f"Arm {e.arm} home, torque off")
            else:
                sc.feed.insert(0, f"Arm {e.arm} did NOT park: {d.get('error', '')}")
        elif e.kind == "plan" and e.task and e.arm and e.plan_version is not None:
            plan_by_version.setdefault(e.plan_version, []).append(
                Bar(e.task, e.arm, e.planned_start or 0.0, e.planned_end or 0.0, "plan")
            )
        elif e.kind == "task_start" and e.task:
            starts[e.task] = e
        elif e.kind == "task_end" and e.task:
            starts.pop(e.task, None)
            if e.arm and "start" in d:
                sc.actual.append(
                    Bar(
                        e.task,
                        e.arm,
                        d["start"],
                        d.get("end", d["start"]),
                        "actual",
                        e.outcome or "",
                        d.get("skill", ""),
                    )
                )
            if e.outcome and e.outcome != "ok":
                why = d.get("reason", e.outcome)
                where = f" on {e.arm}" if e.arm else ""
                sc.feed.insert(0, f"{e.task} {e.outcome}{where} ({why})")
        elif e.kind == "replan":
            reason = d.get("reason", "")
            if reason == "go":
                t0 = d.get("t0", e.t)
                sc.phase = "running"
                sc.feed.insert(0, "Go")
                continue
            if reason == "finished":
                sc.phase = "finished"
                sc.finish = d.get("finish")
                sc.feed.insert(0, f"Finished in {clock_text(sc.finish)}")
                continue
            plan_version = e.plan_version
            if sc.phase in ("empty", "estimate"):
                sc.phase = "estimate"
            sc.finish = d.get("makespan")
            sc.lower_bound = d.get("lower_bound")
            sc.binding = d.get("binding", "")
            sc.blocked = list(d.get("blocked", []))
            sc.whatif = {int(k): float(v) for k, v in d.get("whatif", {}).items()}
            sc.done, sc.total = d.get("done", sc.done), d.get("total", sc.total)
            if sc.phase == "estimate":
                sc.estimate = d.get("makespan")
            if d.get("unblocked"):
                n = len(d["unblocked"])
                sc.unblocked_note = (
                    f"Arm {d.get('arm')} made {n} more step{'s' * (n != 1)} possible."
                )
            if reason in ("arm_added", "arm_removed", "arm_failed") and sc.phase == "running":
                sc.feed.insert(0, f"Re-split: finish now {clock_text(d.get('makespan'))}")
            if reason in REPLAN_REASONS and sc.phase == "running":
                sc.replanning = {
                    "t": e.t,
                    "what": REPLAN_REASONS[reason],
                    "arm": d.get("arm", ""),
                    "now": d.get("now", 0.0),
                    "before": d.get("previous_makespan"),
                    "after": d.get("makespan"),
                    "considered": d.get("considered", {}),
                    "solver": d.get("solver", ""),
                    "arms": len(d.get("arms", {})),
                    "unblocked": len(d.get("unblocked", [])),
                }
        elif e.kind == "scene":
            sc.scene = d
    if plan_version is not None:
        sc.plan = plan_by_version.get(plan_version, [])
    if t0 is not None:
        sc.now = (clock_now if clock_now is not None else last_t) - t0
    if sc.phase == "running":
        for tid, s in starts.items():
            start = s.data.get("start", 0.0)
            sc.actual.append(
                Bar(tid, s.arm or "", start, max(start, sc.now), "running", "", s.data.get("skill"))
            )
    if sc.replanning is not None:
        age = (clock_now if clock_now is not None else last_t) - sc.replanning["t"]
        if sc.phase != "running" or age > REPLAN_HOLD_S:
            sc.replanning = None
        else:
            sc.replanning["age"] = age
    if sc.phase == "finished":
        sc.done = sc.total
    sc.limit = limit_text(sc.binding)
    sc.feed = sc.feed[:8]
    return sc


def replan_text(r: dict) -> tuple[str, str]:
    """Headline and detail for the re-planning moment."""
    arm = f"Arm {r['arm']} {r['what']}" if r.get("arm") else r["what"].capitalize()
    if r.get("age", 0.0) < 1.5:  # first beat: the system is working on it
        n = r.get("arms", 0)
        detail = (
            f"comparing plans for {n} arm{'s' if n != 1 else ''}: reach, free arms, shared areas"
        )
        return f"{arm}: re-planning…", detail
    now = r.get("now", 0.0)
    before, after = r.get("before"), r.get("after")
    plans = ", ".join(
        f"{PLANNER_NAMES.get(k, k)} {clock_text(v - now)}"
        for k, v in sorted(r.get("considered", {}).items(), key=lambda kv: -kv[1])
    )
    head = f"{arm}: new plan"
    if before is not None and after is not None:
        head = f"{arm}: {clock_text(before - now)} left → {clock_text(after - now)} left"
    detail = f"plans compared: {plans}" if plans else ""
    if r.get("unblocked"):  # more work became possible, so the time may go up, not down
        n = r["unblocked"]
        detail = f"{n} more step{'s' if n != 1 else ''} now within reach. " + detail
    return head, detail
