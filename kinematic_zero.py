"""Kinematic zero for SO-101 arms: convert LeRobot joint readings to the angles of the MuJoCo Menagerie SO-101 model
(robotstudio_so101) and back, so IK, FK and simulated arms agree with the real ones.

Why: LeRobot's calibration puts each joint's zero at the middle of its calibrated range. On Vato's arm the elbow's
true kinematic zero (the one the Menagerie model uses) is 6.9 deg away from LeRobot's zero. That offset was measured
with a camera-based kinematic calibration and agrees with three independent estimates (7.3, 8.0 and 8.1 deg). The
other joints were within about 1 deg and are left at 0.

Other arms: LeRobot's zero is the midpoint of the joint's mechanical range, so the same offset should hold on any
SO-101 built from the same parts (inference, not checked on a second arm). Use DEFAULT_OFFSETS_DEG unless an arm
shows a consistent elbow error, then give that arm its own offsets (ArmFrame(offsets_deg=...)).

Convention (LeRobot 0.6, use_degrees=True, the default):
    deg_lerobot = sign * deg_kinematic + offset_deg      (body joints, degrees)
    gripper: LeRobot 0..100 maps linearly onto the Menagerie gripper range (inference, not checked)

Usage:
    from kinematic_zero import ArmFrame
    frame = ArmFrame()                       # Vato's measured offsets
    q = frame.to_kinematic(robot.get_observation())       # np.array, rad, Menagerie joint order
    robot.send_action(frame.to_lerobot(q_target))          # dict of "<joint>.pos"
    python kinematic_zero.py                 # self-test
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BODY = JOINTS[:5]

# Menagerie robotstudio_so101/so101.xml joint ranges [rad]
LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.7453293, 1.7453293),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.658063, 1.658063),
    "wrist_roll": (-2.7438473, 2.8412063),
    "gripper": (-0.174533, 1.7453292),
}

# Measured on Vato's SO-101 (12 V STS3215, LeRobot calibration): elbow 6.9 deg; other joints within ~1 deg, left at 0.
DEFAULT_OFFSETS_DEG = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 6.9, "wrist_flex": 0.0,
                       "wrist_roll": 0.0}
DEFAULT_SIGNS = {j: 1 for j in BODY}


@dataclass
class ArmFrame:
    """LeRobot <-> kinematic (Menagerie) joint angles for one arm."""

    offsets_deg: dict = field(default_factory=lambda: dict(DEFAULT_OFFSETS_DEG))
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(offsets_deg={**DEFAULT_OFFSETS_DEG, **d.get("offsets_deg", {})},
                   signs={**DEFAULT_SIGNS, **d.get("signs", {})})

    def save(self, path):
        Path(path).write_text(json.dumps({"offsets_deg": self.offsets_deg, "signs": self.signs}, indent=1))

    def to_kinematic(self, obs):
        """LeRobot observation or action dict ("<joint>.pos": deg, gripper 0..100) -> rad array in JOINTS order."""
        q = np.empty(len(JOINTS))
        for k, j in enumerate(BODY):
            q[k] = np.radians((obs[f"{j}.pos"] - self.offsets_deg[j]) / self.signs[j])
        lo, hi = LIMITS_RAD["gripper"]
        q[5] = lo + (hi - lo) * obs["gripper.pos"] / 100.0
        return q

    def to_lerobot(self, q, clip=True):
        """rad array in JOINTS order (5 body joints, or 6 with the gripper) -> LeRobot action dict."""
        q = np.asarray(q, float)
        out = {}
        for k, j in enumerate(BODY):
            qk = np.clip(q[k], *LIMITS_RAD[j]) if clip else q[k]
            out[f"{j}.pos"] = float(self.signs[j] * np.degrees(qk) + self.offsets_deg[j])
        if len(q) > 5:
            lo, hi = LIMITS_RAD["gripper"]
            out["gripper.pos"] = float(np.clip((q[5] - lo) / (hi - lo) * 100.0, 0.0, 100.0))
        return out


def _self_test():
    f = ArmFrame()
    obs = {"shoulder_pan.pos": 10.0, "shoulder_lift.pos": -20.0, "elbow_flex.pos": 6.9, "wrist_flex.pos": 30.0,
           "wrist_roll.pos": 0.0, "gripper.pos": 50.0}
    q = f.to_kinematic(obs)
    assert abs(q[2]) < 1e-12, "LeRobot elbow 6.9 deg is the kinematic zero"
    back = f.to_lerobot(q)
    assert all(abs(back[k] - obs[k]) < 1e-9 for k in obs), back
    print("ok: round trip exact; LeRobot elbow 6.9 deg = kinematic 0; offsets", f.offsets_deg)


if __name__ == "__main__":
    _self_test()
