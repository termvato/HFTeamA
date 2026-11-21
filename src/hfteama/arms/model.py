"""The vendored SO-101 MuJoCo model (so101-nexus copy of Menagerie robotstudio_so101).

Shared by the IK solver and the simulated arms, so both agree on the arm's geometry.
"""

from __future__ import annotations

from pathlib import Path

MODEL_XML = Path(__file__).parent / "assets" / "so101" / "so101.xml"
TCP_SITE = "gripperframe"  # tool centre point; local z points from the wrist to the fingertips
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
