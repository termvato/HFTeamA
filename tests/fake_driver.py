"""In-memory ArmDriver for skill and worker tests: records calls, injects failures, no MuJoCo."""

from __future__ import annotations

import threading
import time

import numpy as np

from hfteama.arms.primitives import ArmDown, Grasp, MotionError


class FakeDriver:
    """Records every primitive call in `calls`.

    fail_grasp:      close_gripper reports an empty grasp.
    drop_after_lift: the part is lost on the first move_to after a successful grasp.
    down_on:         name of a method that raises ArmDown("injected") from then on.
    hang_on:         name of a method that blocks until stop(), then raises MotionError("stopped").
    """

    def __init__(
        self,
        arm_id: str = "arm_a",
        move_s: float = 0.0,
        table_z: float = 0.0,
        fail_grasp: bool = False,
        drop_after_lift: bool = False,
        down_on: str | None = None,
        hang_on: str | None = None,
    ) -> None:
        self.arm_id = arm_id
        self.table_z = table_z
        self.move_s = move_s
        self.fail_grasp = fail_grasp
        self.drop_after_lift = drop_after_lift
        self.down_on = down_on
        self.hang_on = hang_on
        self.calls: list[tuple[str, tuple]] = []
        self._held = False
        self._stopped = threading.Event()

    def _enter(self, name: str, *args: object) -> None:
        self.calls.append((name, args))
        if self.down_on == name:
            raise ArmDown("injected")
        if self.hang_on == name:
            self._stopped.clear()
            self._stopped.wait(timeout=30.0)
            raise MotionError("stopped")
        if self.move_s:
            time.sleep(self.move_s)

    def read_joints(self) -> np.ndarray:
        return np.zeros(6)

    def set_joint_target(self, q: np.ndarray) -> None:
        self._enter("set_joint_target", tuple(q))

    def move_joints(self, q: np.ndarray, speed: float = 1.0) -> None:
        self._enter("move_joints", tuple(q), speed)

    def move_to(self, xyz: np.ndarray, speed: float = 1.0) -> None:
        self._enter("move_to", tuple(float(v) for v in xyz))
        if self.drop_after_lift and self._held:
            self._held = False

    def open_gripper(self) -> None:
        self._enter("open_gripper")
        self._held = False

    def close_gripper(self) -> Grasp:
        self._enter("close_gripper")
        self._held = not self.fail_grasp
        return Grasp(holding=self._held, position=0.4 if self._held else -0.1)

    def holding(self) -> bool:
        self._enter("holding")
        return self._held

    def stop(self) -> None:
        self.calls.append(("stop", ()))
        self._stopped.set()

    def park(self) -> None:
        self._enter("park")

    def alive(self) -> float:
        return time.time()
