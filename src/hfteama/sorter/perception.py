"""The sorter's calibrated vision behind Operations' detector interface (perception, #21 / #19).

`perception.live` finds the boxes with a detector (`detect(rgb) -> [Detection]`) and turns each
detection into the table frame with its own marker calibration. `SorterTagDetector` gives the
boxes the sorter grasps on the real arms: the camera pose from the 4 reference tags with the
calibrated C922 intrinsics (lens distortion included), each box's centre cast at its tag height,
and its yaw from the tag (`hfteama.sorter.run.Eye`). The pose handed back is built with the
pipeline's own calibration (`calib.to_camera`), so its `calib.to_table` returns exactly the
sorter's position, whatever that calibration's error.

    HFTEAMA_DETECTOR=sorter python -m hfteama.demo --backend sorter ...
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SorterTagDetector:
    """`detect(rgb)` -> perception Detections of the box tags, positions from the sorter."""

    def __init__(self, calib: Any, sorter_config: Any = None):
        from hfteama.sorter.bridge import MainFrame
        from hfteama.sorter.geometry import load_config
        from hfteama.sorter.run import Eye

        self.calib = calib
        self.cfg = load_config(sorter_config)
        self.frame = MainFrame.from_config(self.cfg)
        self.eye = Eye(self.cfg, image=None)  # no camera opened: frames come from detect()
        self.dropped: list[tuple[int, str]] = []  # (tag id, why), as TagDetector

    def detect(self, rgb: Any, depth: Any = None) -> list[Any]:
        from hfteama.perception.scene import Detection
        from hfteama.perception.tags import tag_key
        from hfteama.vision.table import project

        bgr = np.ascontiguousarray(np.asarray(rgb)[..., ::-1])
        # boxes already in the bowl too: the pipeline's place check looks for them there, and
        # its planner leaves boxes inside a bin alone
        _, pose, scene = self.eye.look_frame(bgr, keep_in_box=True)
        pc = self.cfg["pieces"]
        z = pc.get("grasp_z", 0.015)  # what `hfteama sort export` gives the skills
        rot = self.frame.yaw_to_sorter(0.0)  # sorter yaw = main yaw + rot
        self.dropped = []
        out = []
        for p in scene.pieces:
            if p.tag_id is None:
                continue  # an untagged blob: not a box
            x, y, _ = self.frame.to_main([p.x, p.y, 0.0])
            # Piece.yaw is the jaws' closing direction (sorter frame); a SceneObject's yaw is
            # the box's long side (perception), in main's frame
            yaw = p.yaw - rot - np.pi / 2
            c, s = np.cos(yaw), np.sin(yaw)
            T = np.eye(4)
            T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
            T[:3, 3] = (x, y, z)
            u, v = project([[p.x, p.y, pc.get("object_height", 0.04)]], pose, self.eye.intr)[0]
            out.append(
                Detection(
                    T_cam_obj=self.calib.to_camera(T),
                    model="box",
                    pixel=(float(u), float(v)),
                    label=f"box with tag {p.tag_id}",
                    key=tag_key(p.tag_id),
                    has_yaw=True,
                )
            )
        return out
