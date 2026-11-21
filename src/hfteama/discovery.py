"""Plug and produce: notice arms being plugged in or unplugged and tell the executor.

Each follower arm is its own USB serial adapter with a unique serial number. The watcher polls the
connected serial devices (pyserial, works on Linux, macOS and Windows: no /dev paths), and for
every known serial that appears it builds that arm's worker and calls `executor.add_arm(worker)`;
when a serial disappears it calls `executor.arm_lost(arm_id)`.

Known arms come from Hardware's `configs/teleop_pairs.json` (follower serial -> follower id, which
is also the planner's catalog id). How a worker is built is the caller's business (`factory`):
the real demo passes Action's real worker, tests pass a fake arm. Building a real worker connects
to the arm and may move it to its start pose, so it runs outside the watcher's lock and any error
is reported instead of raised.

A manual fallback for the stage: `plug(arm_id)` / `unplug(arm_id)` do the same as a USB event.

Stale entries (Windows): Windows sometimes misses an unplug and keeps listing the board, with no
device behind it (seen 21 Nov: COM3 listed as present for 15 minutes with nothing plugged in). With
`verify=port_answers` the watcher opens a listed port before it counts the arm as present, at most
every `verify_period_s` per arm: a port that cannot be opened because nothing is there counts as
unplugged; a port held by another program (LeRobot driving the arm, teleop) still counts as present.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Windows error codes that mean "no device behind this port". 5 (access denied = another program has
# it open) is deliberately not here: that port is real.
GONE_WINERRORS = {1, 2, 3, 31, 433, 1167}  # incorrect function, not found, path, not functioning,
# no such device, device not connected
GONE_WORDS = ("Incorrect function", "cannot find the file", "No such file", "No such device")


class _Executor(Protocol):
    def add_arm(self, worker: Any) -> None: ...
    def arm_lost(self, arm_id: str, reason: str = ...) -> None: ...


def connected_serials() -> set[str]:
    """USB serial numbers of the serial devices connected right now."""
    from serial.tools import list_ports

    return {p.serial_number for p in list_ports.comports() if p.serial_number}


def open_error_means_gone(message: str) -> bool:
    """Whether pyserial's 'could not open port' message says the device is not there.

    pyserial puts the Windows error in the text, e.g.
    "could not open port 'COM3': OSError(22, 'Incorrect function.', None, 1)" (stale entry) or
    "... PermissionError(13, 'Access is denied.', None, 5)" (real port, another program has it).
    Anything not recognised counts as there: never drop an arm on a guess.
    """
    m = re.search(r"\(\d+, '[^']*', None, (\d+)\)", message)
    if m:
        return int(m.group(1)) in GONE_WINERRORS
    return any(w in message for w in GONE_WORDS)


def port_answers(serial_number: str) -> bool:
    """Whether the port listed for this USB serial is really there: open it, close it, send nothing.

    DTR and RTS are kept low, so opening does not signal the board. A port another program holds
    (access denied) is there. Meant for Windows, where a port cannot be opened twice; on Linux it
    would open a port another program is using, so the demo does not use it there.
    """
    import serial
    from serial.tools import list_ports

    device = next(
        (p.device for p in list_ports.comports() if p.serial_number == serial_number), None
    )
    if device is None:
        return False
    port = serial.Serial()
    port.port = device
    port.dtr = False
    port.rts = False
    try:
        port.open()
    except (serial.SerialException, OSError) as e:
        return not open_error_means_gone(str(e))
    port.close()
    return True


def followers_from_pairs(path: str | Path) -> dict[str, str]:
    """USB serial -> follower arm id, from Hardware's pairs config (leaders are ignored)."""
    pairs = json.loads(Path(path).read_text())["pairs"]
    out: dict[str, str] = {}
    for pair in pairs.values():
        f = pair.get("follower", {})
        if f.get("serial") and f.get("id"):
            out[f["serial"]] = f["id"]
    return out


class ArmWatcher:
    def __init__(
        self,
        executor: _Executor,
        known: dict[str, str],  # USB serial -> arm id
        factory: Callable[[str], Any],  # arm id -> ArmWorker (may connect to the arm)
        scan: Callable[[], Iterable[str]] = connected_serials,
        period_s: float = 0.5,
        debounce: int = 2,  # scans a serial must be seen (or missing) before we act
        on_error: Callable[[str, Exception], None] | None = None,
        verify: Callable[[str], bool] | None = None,  # USB serial -> really there (port_answers)
        verify_period_s: float = 3.0,  # check each listed serial at most this often
    ) -> None:
        self.executor = executor
        self.known = dict(known)
        self.factory = factory
        self.scan = scan
        self.period_s = period_s
        self.debounce = max(1, debounce)
        self.on_error = on_error
        self.verify = verify
        self.verify_period_s = verify_period_s
        self._verified: dict[str, tuple[float, bool]] = {}  # serial -> (when checked, answered)
        self._ghosts: set[str] = set()  # arm ids listed by the system but not answering
        self._joined: set[str] = set()  # arm ids handed to the executor
        self._held_back: set[str] = set()  # removed on purpose; rejoin only after a replug
        self._count: dict[str, int] = {}  # arm id -> consecutive scans in the new state
        self._present: set[str] = set()  # known arm ids seen in the last successful scan
        self.scanned = False  # True once a scan has run (else `present()` means nothing yet)
        self.errors: list[tuple[str, str]] = []  # (arm id, why) for arms that could not start
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- manual controls (stage fallback), same effect as a USB event

    def plug(self, arm_id: str) -> bool:
        with self._lock:
            if arm_id in self._joined:
                return False
            self._joined.add(arm_id)
        try:
            worker = self.factory(arm_id)
        except Exception as e:  # e.g. the arm refused to start; keep watching the others
            with self._lock:
                self._joined.discard(arm_id)
            log.error("arm %s could not start: %s", arm_id, e)
            with self._lock:
                self.errors = [*self.errors[-19:], (arm_id, str(e) or type(e).__name__)]
            if self.on_error:
                self.on_error(arm_id, e)
            return False
        self.executor.add_arm(worker)
        return True

    def unplug(self, arm_id: str, reason: str = "unplugged") -> bool:
        with self._lock:
            if arm_id not in self._joined:
                return False
            self._joined.discard(arm_id)
        self.executor.arm_lost(arm_id, reason)
        return True

    def forget(self, arm_id: str) -> None:
        """The arm was removed on purpose (slow stop): do not report it as lost when its cable is
        pulled, and let it join again when plugged back in."""
        with self._lock:
            self._joined.discard(arm_id)
            self._held_back.add(arm_id)

    def joined(self) -> set[str]:
        with self._lock:
            return set(self._joined)

    def present(self) -> set[str]:
        """Known arm ids whose USB serial was seen in the last scan (cable in), joined or not."""
        with self._lock:
            return set(self._present)

    # ---- polling

    def ghosts(self) -> set[str]:
        """Known arm ids the system lists but whose port does not answer (stale entries)."""
        with self._lock:
            return set(self._ghosts)

    def _answers(self, serial: str) -> bool:
        if self.verify is None:
            return True
        now = time.monotonic()
        checked = self._verified.get(serial)
        if checked is not None and now - checked[0] < self.verify_period_s:
            return checked[1]
        try:
            ok = bool(self.verify(serial))
        except Exception as e:  # a broken check must not drop an arm
            log.warning("port check for %s failed: %s", serial, e)
            ok = True
        self._verified[serial] = (now, ok)
        return ok

    def poll_once(self) -> None:
        """One scan: act on arms whose presence changed for `debounce` scans in a row."""
        try:
            listed = [s for s in self.scan() if s in self.known]
        except Exception as e:  # a failed scan must not look like every arm was unplugged
            log.warning("serial scan failed: %s", e)
            return
        present = {self.known[s] for s in listed if self._answers(s)}
        ghosts = {self.known[s] for s in listed} - present
        for s in list(self._verified):
            if s not in listed:  # unplugged: check again from scratch when it comes back
                del self._verified[s]
        with self._lock:
            self._present = set(present)
            self._ghosts = ghosts
            self.scanned = True
            self._held_back &= present  # cable pulled: the arm may join again next time
            joined = set(self._joined)
            held = set(self._held_back)
        changed = (present - joined - held) | (joined - present)
        for arm in list(self._count):
            if arm not in changed:
                del self._count[arm]
        for arm in sorted(changed):
            self._count[arm] = self._count.get(arm, 0) + 1
            if self._count[arm] < self.debounce:
                continue
            del self._count[arm]
            if arm in present:
                self.plug(arm)
            elif arm in ghosts:
                self.unplug(arm, "port not answering (listed by the system, nothing there)")
            else:
                self.unplug(arm)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="arm-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.period_s):
            self.poll_once()
