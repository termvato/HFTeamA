"""Perception: the VLM brain of the sorting demo. Owner: Operations (#21), Aravind.

Camera + Go -> categories -> steps; a camera check after each step; a re-scan when the scene
changes. Zero-shot (no training): fixed answer format, validator, retry.

- frames.py      camera frame -> table frame (camera pose in the base frame, intrinsics K)
- pointing.py    VLMPointDetector: the VLM points at objects, pixels projected onto the table
- tags.py        TagDetector: AprilTag/ArUco tags on the boxes give position, yaw and id
- scene.py       detections -> objects with stable ids in the table frame; scene diff
- vlm.py         VLM interface; QwenVLM (Hugging Face transformers); FakeVLM for tests
- prompts.py     decomposer and step-check prompts
- annotate.py    draw object ids on the image for the VLM
- decomposer.py  categories (VLM) -> pick/place steps in the task-graph format on main
- check.py       StepChecker: did this step visibly succeed?
- rescan.py      new / moved / vanished objects -> updated task graph

Output feeds registry.resolve() (allowed_arms, BLOCKED), then the scheduler.
"""

from hfteama.perception.check import CheckResult, StepChecker
from hfteama.perception.decomposer import (
    Bin,
    Category,
    Decomposer,
    Decomposition,
    DecompositionError,
    Sorting,
    graph_to_dict,
    load_bins,
)
from hfteama.perception.frames import Calibration
from hfteama.perception.pointing import DetectionError, VLMPointDetector
from hfteama.perception.rescan import Rescan, rescan
from hfteama.perception.scene import (
    Detection,
    ObjectDetector,
    Scene,
    SceneDiff,
    SceneObject,
    build_scene,
    diff_scenes,
)
from hfteama.perception.tags import (
    TagBox,
    TagConfig,
    TagDetector,
    tag_decomposer,
    tag_sorting,
)
from hfteama.perception.vlm import VLM, FakeVLM, QwenVLM, VLMError

__all__ = [
    "VLM",
    "Bin",
    "Calibration",
    "Category",
    "CheckResult",
    "Decomposer",
    "Decomposition",
    "DecompositionError",
    "DetectionError",
    "Detection",
    "FakeVLM",
    "ObjectDetector",
    "QwenVLM",
    "Rescan",
    "Scene",
    "SceneDiff",
    "SceneObject",
    "TagBox",
    "TagConfig",
    "TagDetector",
    "Sorting",
    "StepChecker",
    "VLMPointDetector",
    "VLMError",
    "build_scene",
    "diff_scenes",
    "graph_to_dict",
    "load_bins",
    "rescan",
    "tag_decomposer",
    "tag_sorting",
]
