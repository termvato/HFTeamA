"""Arm registry: which arms are plugged in, where their bases are, what they can reach.

Owner: Operations (#21, backlog #3).

- The catalog (configs/arms.json) lists every arm we own; an arm joins the registry when Hardware
  detects it plugged in and leaves when it is unplugged or its heartbeat goes stale.
- Positions are in the shared table frame, metres. base_pose_m is the arm base in that frame.
- Writes arm_added / arm_removed / arm_failed to the event log (kinds from events.py).
- Listeners (e.g. the executor) are called on every join / leave / failure, to trigger a re-plan.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from hfteama.arms.base import ArmState, ArmWorker
from hfteama.arms.kinematics import LIMITS_RAD
from hfteama.events import Event, EventLog


@dataclass(frozen=True)
class Reach:
    """Reach envelope in the table plane: a sector of an annulus around the shoulder-pan axis.

    r_min..r_max is measured from the pan axis, which sits pan_offset_m in front of the base
    origin. pan_limit_deg is the hip (shoulder_pan) range either side of the arm's forward
    direction (base +x, rotated by the arm's mounting yaw); 180 = no limit.
    """

    r_min: float = 0.0
    r_max: float = math.inf  # inf = reaches everything ("*" in the config)
    pan_offset_m: float = 0.0
    pan_limit_deg: float = 180.0

    def contains(
        self, base_xy: tuple[float, float], point_xy: tuple[float, float], yaw_deg: float = 0.0
    ) -> bool:
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        ax, ay = base_xy[0] + self.pan_offset_m * c, base_xy[1] + self.pan_offset_m * s
        dx, dy = point_xy[0] - ax, point_xy[1] - ay
        r = math.hypot(dx, dy)
        if not self.r_min <= r <= self.r_max:
            return False
        if self.pan_limit_deg >= 180.0 or r == 0.0:
            return True
        # angle of the point relative to the arm's forward direction, in [-180, 180)
        rel = (math.degrees(math.atan2(dy, dx)) - yaw_deg + 180.0) % 360.0 - 180.0
        return abs(rel) <= self.pan_limit_deg

    @classmethod
    def from_config(cls, value: Any) -> Reach:
        if value in (None, "*"):
            return cls()
        if value == "so101":
            return SO101_REACH
        base = SO101_REACH if value.get("preset") == "so101" else cls()
        return cls(
            r_min=float(value.get("r_min", base.r_min)),
            r_max=float(value.get("r_max", base.r_max)),
            pan_offset_m=float(value.get("pan_offset_m", base.pan_offset_m)),
            pan_limit_deg=float(value.get("pan_limit_deg", base.pan_limit_deg)),
        )


# SO-101 from the MuJoCo Menagerie model (robotstudio_so101), same model as arms/kinematics.py:
# fully extended horizontal reach, pan axis to the gripper frame, is 0.441 m (0.439 m at table
# height); the pan axis is 0.0388 m in front of the base origin; hip range +-110 deg.
SO101_REACH = Reach(
    r_min=0.0,
    r_max=0.44,
    pan_offset_m=0.0388,
    pan_limit_deg=math.degrees(LIMITS_RAD["shoulder_pan"][1]),
)


@dataclass(frozen=True)
class ArmNode:
    arm_id: str
    base_pose_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_yaw_deg: float = 0.0  # mounting yaw: direction of the arm's forward (pan = 0), about z
    reach: Reach = field(default_factory=Reach)
    serial: str = ""
    backend: str = "real"  # "real" | "sim" | "fake", as in arms.base.ArmWorker
    capabilities: frozenset[str] = frozenset()
    state: ArmState = ArmState.STARTING
    last_heartbeat: float = 0.0
    current_task: str | None = None

    @property
    def base_xy(self) -> tuple[float, float]:
        return (self.base_pose_m[0], self.base_pose_m[1])

    @property
    def pan_axis_xy(self) -> tuple[float, float]:
        """Where the hip (shoulder_pan) axis meets the table plane: reach is measured from here."""
        yaw = math.radians(self.base_yaw_deg)
        off = self.reach.pan_offset_m
        return (
            self.base_pose_m[0] + off * math.cos(yaw),
            self.base_pose_m[1] + off * math.sin(yaw),
        )

    def reaches(self, point: tuple[float, ...] | list[float]) -> bool:
        """True if the table-frame point (x, y[, z]) is inside this arm's reach envelope."""
        return self.reach.contains(
            self.base_xy, (float(point[0]), float(point[1])), self.base_yaw_deg
        )


def load_catalog(path: str | Path) -> dict[str, ArmNode]:
    """Every arm we own, from configs/arms.json (see configs/arms.example.json). Not yet joined."""
    raw = json.loads(Path(path).read_text())
    out: dict[str, ArmNode] = {}
    for d in raw["arms"]:
        node = ArmNode(
            arm_id=d["id"],
            base_pose_m=tuple(float(v) for v in d.get("base_pose_m", (0.0, 0.0, 0.0))),
            base_yaw_deg=float(d.get("base_yaw_deg", 0.0)),
            reach=Reach.from_config(d.get("reach", "*")),
            serial=d.get("serial", ""),
            backend=d.get("backend", "real"),
            capabilities=frozenset(d.get("capabilities", ())),
        )
        if node.arm_id in out:
            raise ValueError(f"duplicate arm id {node.arm_id!r} in {path}")
        out[node.arm_id] = node
    return out


def load_real_arms(path: str | Path, reach: Any = "so101") -> dict[str, ArmNode]:
    """The planner's arm catalog from Action's real-arm config (`configs/real_arms.json`), so the
    planner and the real driver read arm positions from one file.

    That file is a dict of arm id -> {"base": [x, y, z] metres in the table frame, "yaw": radians
    about z (0 = the arm's forward is table +x), "pair": ...}; see `arms/real.py::from_config`.
    The arm id must be the worker's `arm_id`. Optional per-arm "reach" overrides `reach`.
    """
    raw = json.loads(Path(path).read_text())["arms"]
    out: dict[str, ArmNode] = {}
    for arm_id, d in raw.items():
        out[arm_id] = ArmNode(
            arm_id=arm_id,
            base_pose_m=tuple(float(v) for v in d.get("base", (0.0, 0.0, 0.0))),
            base_yaw_deg=math.degrees(float(d.get("yaw", 0.0))),
            reach=Reach.from_config(d.get("reach", reach)),
            serial=d.get("serial", ""),
            backend="real",
        )
    return out


Listener = Callable[[str, str], None]  # (event kind, arm_id)


class ArmRegistry:
    """The arms connected right now. Not thread-safe; call it from the executor's loop."""

    def __init__(self, log: EventLog | None = None, heartbeat_timeout: float = 1.5) -> None:
        # 1.5 s = 3 missed heartbeats at 500 ms (architecture doc, section 4.1)
        self.log = log
        self.heartbeat_timeout = heartbeat_timeout
        self._arms: dict[str, ArmNode] = {}
        self._listeners: list[Listener] = []

    # --- queries -------------------------------------------------------------------------
    def __contains__(self, arm_id: object) -> bool:
        return arm_id in self._arms

    def __iter__(self) -> Iterator[ArmNode]:
        return iter(list(self._arms.values()))

    def __len__(self) -> int:
        return len(self._arms)

    def get(self, arm_id: str) -> ArmNode:
        return self._arms[arm_id]

    def live(self) -> list[ArmNode]:
        """Arms that are joined and not down (may be busy or still starting)."""
        return [a for a in self._arms.values() if a.state is not ArmState.DOWN]

    def ready(self) -> list[str]:
        """Ids the scheduler may plan on: READY or BUSY (a busy arm finishes, then takes more)."""
        return [a.arm_id for a in self._arms.values() if a.state in (ArmState.READY, ArmState.BUSY)]

    # --- changes -------------------------------------------------------------------------
    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def join(self, node: ArmNode, now: float | None = None) -> ArmNode:
        """An arm was plugged in (or a sim arm added). Re-joining an id replaces its node."""
        now = time.time() if now is None else now
        node = replace(node, last_heartbeat=now)
        self._arms[node.arm_id] = node
        self._emit(
            "arm_added",
            node.arm_id,
            {"backend": node.backend, "base_pose_m": list(node.base_pose_m)},
            now,
        )
        return node

    def leave(self, arm_id: str, reason: str = "removed", now: float | None = None) -> None:
        """An arm was removed on purpose (after its slow stop)."""
        if self._arms.pop(arm_id, None) is not None:
            self._emit("arm_removed", arm_id, {"reason": reason}, now)

    def fail(self, arm_id: str, reason: str, now: float | None = None) -> None:
        """An arm went down unexpectedly (bus error, heartbeat lost). Kept, marked DOWN."""
        node = self._arms.get(arm_id)
        if node is None or node.state is ArmState.DOWN:
            return
        self._arms[arm_id] = replace(node, state=ArmState.DOWN, current_task=None)
        self._emit("arm_failed", arm_id, {"reason": reason, "task": node.current_task}, now)

    def set_state(self, arm_id: str, state: ArmState, task: str | None = None) -> None:
        node = self._arms[arm_id]
        self._arms[arm_id] = replace(node, state=state, current_task=task)

    def heartbeat(self, arm_id: str, t: float | None = None) -> None:
        node = self._arms[arm_id]
        self._arms[arm_id] = replace(node, last_heartbeat=time.time() if t is None else t)

    def sync(self, worker: ArmWorker) -> None:
        """Copy state and last heartbeat from a worker (arms.base.ArmWorker)."""
        state = worker.state()
        if state is ArmState.DOWN:
            self.fail(worker.arm_id, "worker reported down")
            return
        node = self._arms[worker.arm_id]
        self._arms[worker.arm_id] = replace(node, state=state, last_heartbeat=worker.heartbeat())

    def check_heartbeats(self, now: float | None = None) -> list[str]:
        """Mark arms with a stale heartbeat as failed. Returns their ids."""
        now = time.time() if now is None else now
        stale = [
            a.arm_id
            for a in self._arms.values()
            if a.state is not ArmState.DOWN and now - a.last_heartbeat > self.heartbeat_timeout
        ]
        for arm_id in stale:
            self.fail(arm_id, "heartbeat lost", now)
        return stale

    def _emit(self, kind: str, arm_id: str, data: dict[str, Any], now: float | None) -> None:
        if self.log is not None:
            ev = Event(kind=kind, arm=arm_id, data=data)
            if now is not None:
                ev.t = now
            self.log.write(ev)
        for listener in list(self._listeners):
            listener(kind, arm_id)
