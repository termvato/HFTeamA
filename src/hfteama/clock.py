"""A wall clock that can run faster than real time, for tests and fast dry runs.

With scale=1 it is time.time(). With scale=100, one real second is 100 clock seconds, so fake arms
"take" their full durations while a test finishes quickly. Real arms always use scale=1.
"""

from __future__ import annotations

import threading
import time


class Clock:
    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        self._t0_wall = time.time()
        self._t0_mono = time.monotonic()

    def time(self) -> float:
        """Clock time in seconds (same epoch as time.time())."""
        return self._t0_wall + (time.monotonic() - self._t0_mono) * self.scale

    def sleep(self, seconds: float, stop: threading.Event | None = None) -> bool:
        """Sleep `seconds` of clock time. Returns True if `stop` was set before the time was up."""
        real = max(0.0, seconds) / self.scale
        if stop is None:
            time.sleep(real)
            return False
        return stop.wait(real)

    def real(self, seconds: float) -> float:
        """Real seconds that correspond to `seconds` of clock time."""
        return seconds / self.scale
