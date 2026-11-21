"""Kinematic zero for SO-101 arms: convert LeRobot joint readings to the joint angles of the
MuJoCo Menagerie SO-101 model (robotstudio_so101) and back, so IK, FK and simulated arms agree
with the real ones.

LeRobot's calibration puts each joint's zero at the middle of its calibrated range, which is not
exactly the model's zero. On the arm this was measured on, the elbow differs by 6.9 deg; the other
joints were within about 1 deg and are left at 0. Arms built from the same parts should be close
(inference, not checked on a second arm); if an arm shows a consistent elbow error, give it its own
offsets with ArmFrame(offsets_deg=...).

Convention (LeRobot 0.6, use_degrees=True, the default):
    deg_lerobot = sign * deg_kinematic + offset_deg      (body joints, degrees)
    gripper: LeRobot 0..100 maps linearly onto the Menagerie gripper range (inference, not checked)

Usage:
    from hfteama.arms.kinematics import ArmFrame
    frame = ArmFrame()
    q = frame.to_kinematic(robot.get_observation())   # np.array, rad, Menagerie joint order
    robot.send_action(frame.to_lerobot(q_target))      # dict of "<joint>.pos"
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

# Measured on one SO-101 (12 V STS3215, LeRobot calibration); other joints within ~1 deg.
DEFAULT_OFFSETS_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 6.9,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}
DEFAULT_SIGNS = {j: 1 for j in BODY}


@dataclass
class ArmFrame:
    """LeRobot <-> kinematic (Menagerie) joint angles for one arm."""

    offsets_deg: dict = field(default_factory=lambda: dict(DEFAULT_OFFSETS_DEG))
    signs: dict = field(default_factory=lambda: dict(DEFAULT_SIGNS))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(
            offsets_deg={**DEFAULT_OFFSETS_DEG, **d.get("offsets_deg", {})},
            signs={**DEFAULT_SIGNS, **d.get("signs", {})},
        )

    def save(self, path):
        Path(path).write_text(
            json.dumps({"offsets_deg": self.offsets_deg, "signs": self.signs}, indent=1)
        )

    def to_kinematic(self, obs):
        """LeRobot dict ("<joint>.pos": deg, gripper 0..100) -> rad array in JOINTS order."""
        q = np.empty(len(JOINTS))
        for k, j in enumerate(BODY):
            q[k] = np.radians((obs[f"{j}.pos"] - self.offsets_deg[j]) / self.signs[j])
        lo, hi = LIMITS_RAD["gripper"]
        q[5] = lo + (hi - lo) * obs["gripper.pos"] / 100.0
        return q

    def to_lerobot(self, q, clip=True):
        """rad array in JOINTS order (5 body joints, or 6 with gripper) -> LeRobot action dict."""
        q = np.asarray(q, float)
        out = {}
        for k, j in enumerate(BODY):
            qk = np.clip(q[k], *LIMITS_RAD[j]) if clip else q[k]
            out[f"{j}.pos"] = float(self.signs[j] * np.degrees(qk) + self.offsets_deg[j])
        if len(q) > 5:
            lo, hi = LIMITS_RAD["gripper"]
            out["gripper.pos"] = float(np.clip((q[5] - lo) / (hi - lo) * 100.0, 0.0, 100.0))
        return out
