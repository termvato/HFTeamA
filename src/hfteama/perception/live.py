"""The camera + VLM pipeline behind the demo runner's hooks (Operations, #21).

    python -m hfteama.demo --backend real --table configs/hacklab_table.json \\
        --planner hfteama.perception.live:plan_from_camera \\
        --verifier hfteama.perception.live:verifier \\
        --rescan hfteama.perception.live:rescan_hook [--rescan-every 20]

- planner: camera -> marker calibration -> objects (tags or Qwen pointing) -> categories
  (Decomposer, or the tag config) -> `registry.resolve` on the table's arms. Returns the resolved
  graph; pick steps carry `object`, `label`, `pick_xyz` and, when measured, `pick_yaw_deg`.
- verifier: the executor's step check (the demo applies it to place steps only). With tags it
  just looks whether the box's tag is now inside the target bin (no GPU); else Qwen answers.
- re-scan: a fresh frame -> `perception.rescan` (new, moved, gone, dropped objects) against the
  executor's done / running steps -> resolved graph. Each scan writes a `scene` event, so the
  screen's top view follows. The executor calls it after a lost part; `--rescan-every` also
  calls it periodically and hands the result to `executor.update_graph`.

The hooks take no arguments, so they are configured by environment variables:

    HFTEAMA_TABLE      table layout (default configs/hacklab_table.json)
    HFTEAMA_CAMERA     camera device name (default: "camera.name" in the table file)
    HFTEAMA_IMAGE      a frame file or stream URL instead of the camera (dry runs)
    HFTEAMA_DETECTOR   auto | tags | vlm (auto = tags if the table file has a "tags" section)
    HFTEAMA_TRAVEL     seconds per metre from an arm to its step (default 4.0)
    HFTEAMA_CLEARANCE  metres two arms keep apart where both reach (default 0.15, 0 = off)

Needs the `camera` extra, plus `vlm` unless every box is a listed tag with a bin.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from hfteama.events import Event
from hfteama.executor import PLACE_SKILLS
from hfteama.perception.decomposer import Decomposer, Sorting
from hfteama.perception.rescan import rescan as rescan_scene
from hfteama.perception.scene import Scene, build_scene
from hfteama.registry import load_catalog, resolve
from hfteama.taskgraph import Task, TaskGraph

DETECTORS = ("auto", "tags", "vlm")


class LivePipeline:
    """One camera, one table, one VLM (loaded on first use). Thread-safe: the planner, the
    verifier and the re-scan share the camera and the GPU, so they run one at a time."""

    def __init__(
        self,
        table: str | Path,
        grab: Callable[[], Any],
        detector: str = "auto",
        travel_s_per_m: float | None = 4.0,
        clearance_m: float | None = 0.15,
        vlm: Any = None,
        image_side: int = 1920,
        object_height_m: float = 0.02,
    ) -> None:
        from hfteama.perception.decomposer import load_bins
        from hfteama.perception.tags import TagConfig

        if detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {detector!r}")
        self.table = str(table)
        self.cfg = json.loads(Path(table).read_text())
        if detector == "tags" and "tags" not in self.cfg:
            raise ValueError(f'detector "tags" needs a "tags" section in {table}')
        self.grab = grab
        self.arms = list(load_catalog(table).values())
        self.bins = load_bins(table)
        use_tags = detector == "tags" or (detector == "auto" and "tags" in self.cfg)
        self.tags = TagConfig.from_dict(self.cfg["tags"]) if use_tags else None
        # every box listed with a bin, and only listed boxes counted: no VLM needed at all
        self.no_vlm = (
            self.tags is not None
            and self.tags.only_listed
            and bool(self.tags.boxes)
            and all(b.bin for b in self.tags.boxes.values())
        )
        self.travel_s_per_m = travel_s_per_m or None
        self.clearance_m = clearance_m or None
        self.image_side = image_side
        self.object_height_m = object_height_m
        self._vlm = vlm
        self._lock = threading.RLock()
        self.calib: Any = None
        self._table_markers: tuple[int, ...] = ()
        self.scene: Scene | None = None  # last scan: the `expected` scene of the next re-scan
        self.sorting: Sorting | None = None
        self.source: TaskGraph | None = None  # decomposed, not resolved
        self.executor: Any = None
        self.log: Any = None

    # ------------------------------------------------------------------ pieces

    @property
    def vlm(self) -> Any:
        if self._vlm is None:
            from hfteama.perception.vlm import QwenVLM

            self._vlm = QwenVLM(max_image_side=self.image_side)
        return self._vlm

    def calibrate(self, image: Any) -> Any:
        """Camera pose from the table markers. Done once: the camera is fixed, and later frames
        may have arms over the markers."""
        from hfteama.perception.markers import calibrate_from_markers

        mk = self.cfg["markers"]
        m = calibrate_from_markers(
            np.asarray(image),
            mk["size_m"],
            mk.get("reference_id"),
            dictionary=mk["dictionary"],
            only_ids=mk.get("table_ids"),
        )
        self.calib = m.calib
        self._table_markers = tuple(m.markers)
        return m

    def _detector(self) -> Any:
        if self.tags is not None:
            from hfteama.perception.tags import TagDetector

            same = self.tags.dictionary == self.cfg["markers"]["dictionary"]
            return TagDetector(self.calib, self.tags, self._table_markers if same else ())
        from hfteama.perception.pointing import VLMPointDetector

        return VLMPointDetector(self.vlm, self.calib, object_height_m=self.object_height_m)

    def _decomposer(self) -> Decomposer:
        if self.tags is not None and self.no_vlm:
            from hfteama.perception.tags import tag_decomposer

            return tag_decomposer(self.bins, self.tags)
        return Decomposer(self.vlm, self.bins)

    def observe(self, image: Any, previous: Scene | None = None) -> Scene:
        return build_scene(self._detector().detect(image), self.calib, previous)

    def resolved(self, source: TaskGraph) -> TaskGraph:
        return resolve(
            source, self.arms, travel_s_per_m=self.travel_s_per_m, clearance_m=self.clearance_m
        ).graph

    def scene_data(self, graph: TaskGraph) -> dict[str, Any]:
        from hfteama.demo import table_scene

        return table_scene(graph, self.table, [a.arm_id for a in self.arms])

    def _write_scene(self, graph: TaskGraph) -> None:
        if self.log is not None:
            self.log.write(Event("scene", data=self.scene_data(graph)))

    # ------------------------------------------------------------------ the three hooks

    def plan(self) -> TaskGraph:
        """Go: the whole table -> resolved task graph."""
        with self._lock:
            image = self.grab()
            if self.calib is None:
                self.calibrate(image)
            scene = self.observe(image)
            dec = self._decomposer().decompose(image, scene)
            self.scene, self.sorting, self.source = scene, dec.sorting, dec.graph
            graph = self.resolved(dec.graph)
            self._write_scene(graph)
            return graph

    def verifier(self) -> Callable[[Task, Any], tuple[bool, str]]:
        if self.tags is not None:
            return self._tag_verify
        from hfteama.perception.check import StepChecker

        inner = StepChecker(self.vlm, self.bins).as_verifier(self.grab)

        def verify(task: Task, result: Any) -> tuple[bool, str]:
            with self._lock:
                return inner(task, result)

        return verify

    def _tag_verify(self, task: Task, result: Any) -> tuple[bool, str]:
        """Place: is the box's tag inside the target bin now? Other steps: trusted."""
        if task.skill not in PLACE_SKILLS:
            return True, "not checked"
        bin_ = next((b for b in self.bins if b.id == task.args.get("target")), None)
        oid = task.args.get("object")
        if bin_ is None or oid is None:
            return True, "nothing to check"
        with self._lock:
            try:
                scene = self.observe(self.grab(), self.scene)
            except Exception as e:  # camera unplugged, driver error, ...
                return False, f"camera error: {e}"
        obj = scene.objects.get(oid)
        if obj is None:
            return False, f"tag of {oid} not seen"
        if bin_.contains(obj.xyz):
            return True, f"{oid} in {bin_.id}"
        return False, f"{oid} seen outside {bin_.id}"

    def attach(self, executor: Any, log: Any = None) -> None:
        """The re-scan needs the executor's done / running steps; the log gets `scene` events."""
        self.executor = executor
        self.log = log if log is not None else getattr(executor, "log", None)

    def rescan(self) -> TaskGraph | None:
        """A fresh frame -> updated resolved graph, or None if nothing changed."""
        with self._lock:
            if self.source is None or self.sorting is None or self.scene is None:
                raise RuntimeError("re-scan before the first plan")
            status = self.executor.snapshot()["status"] if self.executor is not None else {}
            done = {t for t, s in status.items() if s == "done"}
            running = {t for t, s in status.items() if s in ("running", "verifying")}
            image = self.grab()
            observed = self.observe(image, self.scene)
            r = rescan_scene(
                self._decomposer(),
                image,
                observed,
                self.scene,
                self.source,
                self.sorting,
                done=done,
                running=running,
            )
            self.scene, self.sorting, self.source = r.scene, r.sorting, r.graph
            graph = self.resolved(r.graph)
            self._write_scene(graph)
            return graph if r.changed else None


def from_env() -> LivePipeline:
    from hfteama.perception.camera import frame_source

    table = os.environ.get("HFTEAMA_TABLE", "configs/hacklab_table.json")
    cam = json.loads(Path(table).read_text()).get("camera", {})
    grab = frame_source(
        device=os.environ.get("HFTEAMA_CAMERA") or cam.get("name"),
        image=os.environ.get("HFTEAMA_IMAGE"),
        size=tuple(cam.get("size", (1920, 1080))),
    )
    return LivePipeline(
        table,
        grab,
        detector=os.environ.get("HFTEAMA_DETECTOR", "auto"),
        travel_s_per_m=float(os.environ.get("HFTEAMA_TRAVEL", "4.0")),
        clearance_m=float(os.environ.get("HFTEAMA_CLEARANCE", "0.15")),
    )


_pipeline: LivePipeline | None = None
_pipeline_lock = threading.Lock()


def pipeline() -> LivePipeline:
    """The process-wide pipeline the hooks share (built from the environment on first use)."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = from_env()
        return _pipeline


def plan_from_camera() -> TaskGraph:
    """--planner hook."""
    return pipeline().plan()


def verifier() -> Callable[[Task, Any], tuple[bool, str]]:
    """--verifier hook."""
    return pipeline().verifier()


def rescan_hook(executor: Any, log: Any = None) -> Callable[[], TaskGraph | None]:
    """--rescan hook: Executor(rescan=...) after a lost part, and the periodic re-scan."""
    p = pipeline()
    p.attach(executor, log)
    return p.rescan
