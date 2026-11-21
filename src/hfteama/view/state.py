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
                sc.feed.insert(0, f"Arm {e.arm} parked and removed")
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
    if sc.phase == "finished":
        sc.done = sc.total
    sc.limit = limit_text(sc.binding)
    sc.feed = sc.feed[:8]
    return sc
