"""Qwen as the VLM layer over the sorter's camera (Operations, #21; architecture: VLM in front).

The sorter finds the boxes by their tags (exact position and yaw, ~10 ms). The VLM does what tags
cannot: it looks at the same frame, with each box's id drawn on it, names what it sees and decides
the categories and which bin each box goes to (`Decomposer.categorize`). During the run it
confirms each place from a fresh frame (`StepChecker`, place steps only, decided 21 Nov).

    sort_boxes(vlm, frame_bgr, boxes, bins) -> {box id: {"category", "rule", "bin"}}
    place_checker(vlm, bins, grab) -> verifier(task, result) for Executor(verifier=...)

Geometry never comes from the VLM here, so a wrong answer can cost a category, not a grasp.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from hfteama.perception.decomposer import Bin, Decomposer
from hfteama.perception.scene import Scene, SceneObject


@dataclass(frozen=True)
class SeenBox:
    id: str  # task-graph object id, e.g. "box_tag2"
    pixel: tuple[float, float]  # its centre in the frame
    label: str = "box"  # what the detector knows (shown to the VLM next to the id)


def bgr_to_rgb(frame: Any) -> Any:
    """OpenCV frames are BGR; the VLM wants RGB."""
    return frame[..., ::-1].copy()


def sort_boxes(
    vlm: Any, frame_bgr: Any, boxes: Sequence[SeenBox], bins: Sequence[Bin]
) -> tuple[dict[str, dict[str, str]], float]:
    """Ask the VLM for categories and bins of `boxes`. Returns ({id: {category, rule, bin}},
    seconds). Raises perception.DecompositionError if no valid answer after the retries."""
    if not boxes:
        return {}, 0.0
    scene = Scene(
        objects={
            b.id: SceneObject(
                b.id, "box", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), b.pixel, 1.0, b.label
            )
            for b in boxes
        }
    )
    t0 = time.monotonic()
    sorting, _ = Decomposer(vlm, list(bins)).categorize(
        bgr_to_rgb(frame_bgr), scene, [b.id for b in boxes]
    )
    out = {
        oid: {
            "category": cat,
            "rule": sorting.categories[cat].rule,
            "bin": sorting.categories[cat].bin,
        }
        for oid, cat in sorting.assignment.items()
    }
    return out, time.monotonic() - t0


def place_checker(
    vlm: Any, bins: Sequence[Bin], grab: Callable[[], Any], threshold: float = 0.6
) -> Callable[[Any, Any], tuple[bool, str]]:
    """The executor's step check over the overhead camera: `grab()` returns a BGR frame."""
    from hfteama.perception.check import StepChecker

    checker = StepChecker(vlm, list(bins), threshold=threshold)
    return checker.as_verifier(lambda: bgr_to_rgb(grab()))
