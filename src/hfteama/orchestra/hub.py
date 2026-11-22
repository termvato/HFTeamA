"""StateHub: everything the Orchestra screen shows, from the running demo process.

Two sources (docs/ORCHESTRA_DASHBOARD.md, section 1):
- the event log, the record of what happened, read incrementally and turned into a
  `view.state.Screen` (countdown, limit, what-if, plan against actual, table map);
- the present: `Executor.snapshot()`, each worker's `state()` and `heartbeat()`, and which USB
  serials the `ArmWatcher` saw in its last scan.

Read only. It never calls a method that changes the executor, the watcher or an arm.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from hfteama.events import Event, EventLog
from hfteama.view.state import Bar, Screen, build, replan_text

# Arm colours for the screen, by the word the table config uses for the arm ("black arm, ...").
COLOURS = {
    "black": "#30343a",
    "green": "#2e8b57",
    "blue": "#3b6fb6",
    "red": "#b3403a",
    "white": "#9aa3ab",
    "orange": "#c77c2e",
    "yellow": "#b89b2a",
    "grey": "#6b7680",
}
PALETTE = ["#3b6fb6", "#8a5cb8", "#c77c2e", "#2a8a8a", "#b3403a"]


def pairs_info(path: str | Path) -> dict[str, dict[str, str]]:
    """Follower arm id -> {pair, serial}, from Hardware's configs/teleop_pairs.json."""
    pairs = json.loads(Path(path).read_text())["pairs"]
    out: dict[str, dict[str, str]] = {}
    for name, pair in pairs.items():
        f = pair.get("follower", {})
        if f.get("id"):
            out[f["id"]] = {"pair": name, "serial": f.get("serial", "")}
    return out


def table_info(path: str | Path | None) -> dict[str, dict[str, str]]:
    """Arm id -> {label, colour} from a table layout (configs/hacklab_table.json), if given."""
    if not path:
        return {}
    out: dict[str, dict[str, str]] = {}
    for a in json.loads(Path(path).read_text()).get("arms", []):
        desc = a.get("description", "")
        word = desc.split(" ", 1)[0].lower() if desc else ""
        label = desc.replace(" in the top camera", "")
        out[a["id"]] = {"label": label[:1].upper() + label[1:], "color": COLOURS.get(word, "")}
    return out


def steps_info(graph: Any) -> list[dict[str, Any]]:
    """The batch's steps for the screen: id, skill, object, a readable label, target, estimate."""
    labels: dict[str, str] = {}
    for t in graph.tasks.values():
        obj = t.args.get("object")
        if obj and t.args.get("label"):
            labels[obj] = t.args["label"]
    out = []
    for t in graph.tasks.values():
        obj = t.args.get("object")
        target = t.args.get("target") or t.args.get("to")
        out.append(
            {
                "id": t.id,
                "skill": t.skill,
                "object": obj,
                "label": labels.get(obj, (obj or t.id).replace("_", " ")),
                "target": target,
                "estimate": t.duration,
            }
        )
    return out


def schedule(sc: Screen) -> dict[str, Any]:
    """Plan against actual for the Gantt: the current plan's bars, every finished step, and the
    steps running now (their end is 'now'). Run seconds, rounded so an idle state stays equal."""

    def bar(b: Bar) -> dict[str, Any]:
        out = {"task": b.task, "arm": b.arm, "start": round(b.start, 2), "end": round(b.end, 2)}
        if b.kind != "plan":
            out["state"] = "running" if b.kind == "running" else (b.outcome or "ok")
        return out

    return {
        "now": round(sc.now, 2),
        "finish": sc.finish if sc.phase in ("running", "finished") else None,
        "plan": [bar(b) for b in sc.plan],
        "actual": [bar(b) for b in sc.actual],
    }


class StateHub:
    def __init__(
        self,
        log: EventLog,
        executor: Any,
        watcher: Any,
        rig: Any,
        clock: Any,
        backend: str,
        pairs_path: str | Path,
        table_path: str | Path | None = None,
        watching: bool = False,
        task_name: str = "",
    ) -> None:
        self.log = log
        self.ex = executor
        self.watcher = watcher
        self.rig = rig
        self.clock = clock
        self.backend = backend
        self.watching = watching
        self.task_name = task_name
        self.pairs = pairs_info(pairs_path)
        self.table = table_info(table_path)
        self.arm_ids = list(rig.known)
        self._events: list[Event] = []
        self._offset = 0
        self._lock = threading.Lock()
        self.notes: list[dict[str, Any]] = []  # command lines for the feed, newest first

    # ------------------------------------------------------------------ event log

    def _tail(self) -> None:
        """Read complete new lines of the event log (a half-written last line waits)."""
        path = self.log.path
        if not path.exists():
            return
        with path.open("rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        for line in chunk[: end + 1].decode("utf-8").splitlines():
            if line.strip():
                self._events.append(Event(**json.loads(line)))
        self._offset += end + 1

    def events(self, after: int = 0) -> list[tuple[int, Event]]:
        """Event-log lines after line number `after` (1-based numbers), for the stream."""
        with self._lock:
            self._tail()
            return [(i + 1, e) for i, e in enumerate(self._events) if i + 1 > after]

    _port_cache: tuple[float, dict[str, str]] = (0.0, {})

    def _ports(self) -> dict[str, str]:
        """USB serial -> port name (COM3, /dev/ttyACM0), at most every 2 s. Display only."""
        at, ports = self._port_cache
        if time.monotonic() - at < 2.0:
            return ports
        try:
            from serial.tools import list_ports

            ports = {p.serial_number: p.device for p in list_ports.comports() if p.serial_number}
        except Exception:  # no pyserial, or the scan failed: the screen just shows no port
            ports = {}
        self._port_cache = (time.monotonic(), ports)
        return ports

    def note(self, text: str, level: str = "info") -> None:
        with self._lock:
            self.notes = [{"text": text, "level": level, "wall": self.clock.time()}, *self.notes][
                :10
            ]

    # ------------------------------------------------------------------ config and state

    def config(self, commands: dict[str, bool]) -> dict[str, Any]:
        return {
            "server": "hfteama.orchestra",
            "backend": self.backend,
            "watching": self.watching,
            "catalog": self.arm_ids,
            "task": {
                "name": self.task_name or self.ex.graph.name,
                "steps": steps_info(self.ex.graph),
            },
            "commands": commands,
        }

    def tasks(self) -> list[dict[str, Any]]:
        snap = self.ex.snapshot()
        last_arm: dict[str, str] = {}
        for _, e in self.events():
            if e.kind in ("task_start", "task_end") and e.task and e.arm:
                last_arm[e.task] = e.arm
        running = {t: a for a, t in snap["busy"].items()}
        return [
            {
                "id": t,
                "state": s,
                "arm": running.get(t) or last_arm.get(t),
                "retries": snap["retries"].get(t, 0),
            }
            for t, s in snap["status"].items()
        ]

    def state(self) -> dict[str, Any]:
        evs = [e for _, e in self.events()]
        now_wall = self.clock.time()
        sc = build(evs, clock_now=now_wall)
        snap = self.ex.snapshot()

        t0 = next(
            (
                e.data.get("t0", e.t)
                for e in evs
                if e.kind == "replan" and e.data.get("reason") == "go"
            ),
            None,
        )
        rel = (lambda t: t - t0) if t0 is not None else (lambda t: None)
        per_arm: dict[str, dict[str, Any]] = {
            a: {
                "ok": 0,
                "failed": 0,
                "last": None,
                "joined_at": None,
                "left_at": None,
                "reason": None,
                "unblocked": 0,
            }
            for a in self.arm_ids
        }
        for e in evs:
            a = per_arm.get(e.arm or "")
            if (
                e.kind == "replan"
                and e.data.get("reason") == "arm_added"
                and e.data.get("arm") in per_arm
            ):
                per_arm[e.data["arm"]]["unblocked"] = len(e.data.get("unblocked", []))
            if a is None:
                continue
            if e.kind == "task_end":
                a["ok" if e.outcome == "ok" else "failed"] += 1
            elif e.kind == "arm_added":
                a.update(last="added", joined_at=rel(e.t), reason=None, left_at=None)
            elif e.kind == "arm_removed":
                a.update(last="removed", left_at=rel(e.t), reason="parked and removed")
            elif e.kind == "arm_failed":
                a.update(last="lost", left_at=rel(e.t), reason=e.data.get("reason", "lost"))

        joined = set(snap["arms"])
        present = self.watcher.present() if self.watching else set(joined)
        ports = self._ports() if self.watching else {}
        errors = dict(self.watcher.errors)
        items = []
        for i, arm in enumerate(self.arm_ids):
            info = per_arm[arm]
            worker = self.rig.workers.get(arm)
            is_in = arm in joined
            busy = snap["busy"].get(arm)
            hb = None
            state_word = None
            if is_in and worker is not None:
                try:
                    hb = max(0.0, now_wall - float(worker.heartbeat()))
                    state_word = getattr(worker.state(), "value", str(worker.state()))
                except Exception:  # a dying arm must not break the screen
                    hb, state_word = None, "down"
            if is_in:
                if arm in snap["parking"]:
                    node = "parking"
                elif busy:
                    node = "busy"
                elif state_word == "starting":
                    node = "starting"
                else:
                    node = "ready"
            elif info["last"] == "lost" and arm not in present:
                node = "lost"
            elif info["last"] == "removed" and arm not in present:
                node = "removed"
            elif arm in errors and arm in present:
                node = "error"
            elif arm in present:
                node = "connected"
            else:
                node = "known"
            run = snap["runs"].get(busy) if busy else None
            colour = self.table.get(arm, {}).get("color") or PALETTE[i % len(PALETTE)]
            items.append(
                {
                    "arm_id": arm,
                    "label": self.table.get(arm, {}).get("label", ""),
                    "color": colour,
                    "pair": self.pairs.get(arm, {}).get("pair"),
                    "serial": self.pairs.get(arm, {}).get("serial"),
                    "port": ports.get(self.pairs.get(arm, {}).get("serial", ""))
                    if arm in present
                    else None,
                    "node_state": node,
                    "arm_state": state_word,
                    "backend": getattr(worker, "backend", self.backend) if worker else self.backend,
                    "connected": arm in present or is_in,
                    "joined": is_in,
                    "parking": arm in snap["parking"],
                    "task": busy,
                    "task_elapsed": (snap["now"] - run["start"]) if run else None,
                    "task_estimate": run["estimate"] if run else None,
                    "held": snap["held"].get(arm),
                    "heartbeat_age_s": hb,
                    "joined_at": info["joined_at"],
                    "left_at": info["left_at"],
                    "steps_ok": info["ok"],
                    "steps_failed": info["failed"],
                    "last_reason": errors.get(arm) if node == "error" else info["reason"],
                    "unblocked": info["unblocked"],
                }
            )

        feed = [
            {"t": None, "text": n["text"], "level": n["level"], "wall": n["wall"]}
            for n in self.notes
        ]
        for text in sc.feed:
            lvl = "bad" if any(w in text for w in ("lost", "failed", "timeout")) else "info"
            feed.append({"t": None, "text": text, "level": lvl, "wall": None})
        replanning = None
        if sc.replanning:
            head, detail = replan_text(sc.replanning)
            replanning = {"head": head, "detail": detail}
        phase = sc.phase
        if not snap[
            "started"
        ]:  # before Go: ready only with an arm in; no arm = nothing to estimate
            phase = "estimate" if joined else "empty"
        return {
            "run": {
                "phase": phase,
                "paused": bool(snap.get("paused")),
                "backend": self.backend,
                "now": sc.now,
                "remaining": sc.remaining,
                "predicted_finish": sc.finish,
                "estimate": sc.estimate if joined or snap["started"] else None,
                "lower_bound": sc.lower_bound,
                "binding": sc.binding,
                "limit": sc.limit,
                "done": sc.done,
                "total": sc.total or len(snap["status"]),
                "blocked": sc.blocked,
                "plan_version": snap["plan_version"],
                "whatif": {str(k): v for k, v in sc.whatif.items()},
                "unblocked_note": sc.unblocked_note,
            },
            "schedule": schedule(sc),
            "scene": sc.scene or None,
            "nodes": {
                "known": len(self.arm_ids),
                "connected": sum(1 for n in items if n["connected"]),
                "joined": len(joined),
                "items": items,
            },
            "replanning": replanning,
            "feed": feed[:12],
        }
