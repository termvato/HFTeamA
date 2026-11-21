"""Send every arm's joint angles to a 3D scene (the Isaac Sim rehearsal, issue #33) while it runs.

One UDP datagram per arm per tick: {"arm": <index>, "id": "<arm id>", "q": [6 angles in radians]},
joint order shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper (the SO-101
model's order). `scripts/isaac/so101_scene.py --listen PORT` follows it. Reading joints is all it
does: it never commands an arm.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Callable, Sequence


class JointStreamer:
    def __init__(
        self,
        readers: dict[str, Callable[[], Sequence[float]]],  # arm id -> current joint angles
        host: str = "127.0.0.1",
        port: int = 5605,
        rate_hz: float = 30.0,
    ) -> None:
        self.readers = dict(readers)
        self.addr = (host, port)
        self.period = 1.0 / rate_hz
        self.sent = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, arm_id: str, reader: Callable[[], Sequence[float]]) -> None:
        self.readers[arm_id] = reader

    def send_once(self) -> None:
        for index, (arm_id, read) in enumerate(list(self.readers.items())):
            try:
                q = [float(x) for x in read()]
            except Exception:  # an arm that cannot be read is skipped, the others keep streaming
                continue
            msg = json.dumps({"arm": index, "id": arm_id, "q": q}).encode()
            try:
                self._sock.sendto(msg, self.addr)
                self.sent += 1
            except OSError:
                pass

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="joint-stream", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._sock.close()

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            self.send_once()
