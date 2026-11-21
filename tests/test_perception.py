"""VLM decomposer, step check and re-scan, on a scripted VLM and synthetic SAM-6D detections."""

import json
from pathlib import Path

import numpy as np
import pytest

from hfteama.arms.base import ArmState
from hfteama.perception import (
    Bin,
    Calibration,
    Decomposer,
    DecompositionError,
    Detection,
    FakeVLM,
    StepChecker,
    build_scene,
    graph_to_dict,
    load_bins,
    rescan,
)
from hfteama.perception.vlm import ParseError, parse_json
from hfteama.registry import ArmNode, Reach, load_catalog, resolve
from hfteama.taskgraph import load

CONFIGS = Path(__file__).parent.parent / "configs"
IDENTITY = Calibration(np.eye(4))
BINS = [
    Bin("bin_1", "the left bin", (0.10, 0.25, 0.05)),
    Bin("bin_2", "the right bin", (0.80, 0.25, 0.05)),
]


def det(x: float, y: float, z: float = 0.02, model: str = "cube", px=(100.0, 100.0)) -> Detection:
    T = np.eye(4)
    T[:3, 3] = (x, y, z)
    return Detection(T_cam_obj=T, model=model, pixel=px)


def answer(cats: dict[str, str], asg: dict[str, str]) -> str:
    return json.dumps(
        {
            "categories": [{"name": n, "rule": f"{n} things", "bin": b} for n, b in cats.items()],
            "assignments": [{"object": o, "category": c} for o, c in asg.items()],
        }
    )


GOOD = answer({"red": "bin_1", "blue": "bin_2"}, {"obj_01": "red", "obj_02": "blue"})


def two_objects():
    return build_scene([det(0.2, 0.0), det(0.6, 0.0)], IDENTITY)


# --- frames and scene ----------------------------------------------------------------------


def test_camera_example_maps_to_table_frame() -> None:
    cal = Calibration.load(
        CONFIGS / "camera.example.json", load_catalog(CONFIGS / "arms.example.json")
    )
    T = np.eye(4)
    T[:3, 3] = (0.05, 0.10, 0.78)  # 78 cm in front of a camera looking straight down
    p = cal.to_table(T)[:3, 3]
    # (0.45, 0.10, 0.02) in arm_a's base frame; arm_a is mounted at yaw 90 deg
    assert p == pytest.approx([-0.10, 0.45, 0.02])


def test_base_yaw_rotates_camera_frame() -> None:
    arm = ArmNode("arm_b", base_pose_m=(1.0, 0.0, 0.0), base_yaw_deg=90.0)
    cal = Calibration.from_base(arm, np.eye(4))
    T = np.eye(4)
    T[:3, 3] = (0.2, 0.0, 0.0)
    assert cal.to_table(T)[:3, 3] == pytest.approx([1.0, 0.2, 0.0])


def test_scene_ids_stay_stable() -> None:
    s1 = build_scene([det(0.2, 0.0), det(0.6, 0.0)], IDENTITY)
    assert s1.ids() == ["obj_01", "obj_02"]
    # second frame: order swapped, small jitter, one new object, one low-score ghost
    ghost = det(0.4, 0.4)
    ghost.score = 0.1
    s2 = build_scene([det(0.61, 0.0), det(0.3, 0.3), det(0.2, 0.01), ghost], IDENTITY, s1)
    assert s2.objects["obj_01"].xyz[:2] == pytest.approx((0.2, 0.01))
    assert s2.objects["obj_02"].xyz[:2] == pytest.approx((0.61, 0.0))
    assert s2.objects["obj_03"].xyz[:2] == pytest.approx((0.3, 0.3))
    assert len(s2.objects) == 3


def test_parse_json_tolerates_wrappers() -> None:
    assert parse_json('<think>hmm {"x": 0}</think>\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"a": 2} done') == {"a": 2}
    with pytest.raises(ParseError):
        parse_json("no json here")


# --- decomposer ----------------------------------------------------------------------------


def test_decompose_builds_pick_place_steps(tmp_path: Path) -> None:
    vlm = FakeVLM([GOOD])
    dec = Decomposer(vlm, BINS, annotate=False)
    d = dec.decompose(None, two_objects())
    assert d.attempts == 1
    g = d.graph
    assert set(g.tasks) == {"pick_obj_01", "place_obj_01", "pick_obj_02", "place_obj_02"}
    place = g.tasks["place_obj_01"]
    assert place.deps == ("pick_obj_01",) and place.same_arm_as == "pick_obj_01"
    assert place.args["target"] == "bin_1" and place.resources == ("bin_1",)
    assert place.args["place_xyz"] == [0.10, 0.25, 0.05]
    assert g.tasks["pick_obj_02"].args["pick_xyz"] == pytest.approx([0.6, 0.0, 0.02])
    assert d.grouping()["red"] == {"rule": "red things", "bin": "bin_1", "objects": ["obj_01"]}
    assert "obj_01" in vlm.calls[0] and "bin_2: the right bin" in vlm.calls[0]

    # the graph is in main's task format: it round-trips through taskgraph.load
    p = tmp_path / "vlm_sort.json"
    p.write_text(json.dumps(graph_to_dict(g)))
    assert load(p).tasks == g.tasks


def test_decompose_retries_with_errors() -> None:
    bad = answer({"red": "bin_1", "also_red": "bin_1"}, {"obj_01": "red", "obj_99": "red"})
    vlm = FakeVLM(["not json at all", bad, GOOD])
    d = Decomposer(vlm, BINS, annotate=False).decompose(None, two_objects())
    assert d.attempts == 3
    assert "no JSON object" in vlm.calls[1]
    assert "bin 'bin_1' is used by" in vlm.calls[2]
    assert "unknown object 'obj_99'" in vlm.calls[2]
    assert "objects not assigned: obj_02" in vlm.calls[2]


def test_decompose_gives_up_after_three_retries() -> None:
    vlm = FakeVLM(["{}"] * 4)
    with pytest.raises(DecompositionError):
        Decomposer(vlm, BINS, annotate=False).decompose(None, two_objects())
    assert len(vlm.calls) == 4


def test_objects_already_in_a_bin_are_skipped() -> None:
    scene = build_scene([det(0.2, 0.0), det(0.11, 0.26)], IDENTITY)  # obj_02 sits in bin_1
    vlm = FakeVLM([answer({"red": "bin_1"}, {"obj_01": "red"})])
    d = Decomposer(vlm, BINS, annotate=False).decompose(None, scene)
    assert d.already_sorted == ["obj_02"]
    assert set(d.graph.tasks) == {"pick_obj_01", "place_obj_01"}


def test_decomposition_feeds_reachability() -> None:
    reach = Reach(0.05, 0.40)
    a = ArmNode("arm_a", (0.0, 0.0, 0.0), reach=reach, state=ArmState.READY)
    b = ArmNode("arm_b", (0.9, 0.0, 0.0), reach=reach, state=ArmState.READY)
    d = Decomposer(FakeVLM([GOOD]), BINS, annotate=False).decompose(None, two_objects())
    # obj_01 (0.2, 0) -> bin_1 (0.1, 0.25): arm_a. obj_02 (0.6, 0) -> bin_2 (0.8, 0.25): arm_b.
    r = resolve(d.graph, [a, b])
    assert r.graph.tasks["pick_obj_01"].allowed_arms == ("arm_a",)
    assert r.graph.tasks["place_obj_02"].allowed_arms == ("arm_b",)
    only_a = resolve(d.graph, [a])
    assert only_a.blocked == {"pick_obj_02", "place_obj_02"}


def test_label_objects_draws_on_image() -> None:
    pytest.importorskip("PIL")
    from hfteama.perception.annotate import label_objects

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    out = label_objects(img, two_objects())
    assert out.size == (200, 200) and np.asarray(out).any()


def test_example_bins_load() -> None:
    assert [b.id for b in load_bins(CONFIGS / "bins.example.json")] == ["bin_1", "bin_2", "bin_3"]


# --- step check ----------------------------------------------------------------------------


def _place_task():
    d = Decomposer(FakeVLM([GOOD]), BINS, annotate=False).decompose(None, two_objects())
    return d.graph.tasks["place_obj_01"]


def test_check_confirms_and_doubts() -> None:
    task = _place_task()
    vlm = FakeVLM(
        [
            '{"done": true, "confidence": 0.9, "reason": "in bin"}',
            '{"done": true, "confidence": 0.3}',
            '{"done": false, "confidence": 0.8, "reason": "on the floor"}',
        ]
    )
    chk = StepChecker(vlm, BINS)
    assert chk.check(task, None).done
    unsure = chk.check(task, None)
    assert not unsure.done and unsure.reason.startswith("not sure")
    assert chk.check(task, None).reason == "on the floor"
    assert "Is the red object obj_01 now inside the left bin?" in vlm.calls[0]


def test_check_fails_safe_on_garbage() -> None:
    vlm = FakeVLM(["maybe?", '{"done": "yes"}'])
    r = StepChecker(vlm, BINS, max_retries=1).check(_place_task(), None)
    assert not r.done and r.reason.startswith("check_unparsed")
    assert len(vlm.calls) == 2


# --- re-scan -------------------------------------------------------------------------------


def test_rescan_follows_scene_changes() -> None:
    s0 = build_scene([det(0.2, 0.0), det(0.6, 0.0), det(0.3, -0.1)], IDENTITY)
    go = answer(
        {"red": "bin_1", "blue": "bin_2"}, {"obj_01": "red", "obj_02": "blue", "obj_03": "red"}
    )
    vlm = FakeVLM([go, answer({"red": "bin_1"}, {"obj_05": "red"})])
    dec = Decomposer(vlm, BINS, annotate=False)
    d = dec.decompose(None, s0)

    # obj_01 is being picked: lifted in the gripper (too far to match, gets id obj_04, and is
    # ignored as held). obj_02 moved 4 cm. obj_03 vanished. A new object appears (obj_05).
    s1 = build_scene(
        [det(0.2, 0.0, z=0.15), det(0.64, 0.0), det(0.5, 0.3, px=(300.0, 50.0))], IDENTITY, s0
    )
    r = rescan(dec, None, s1, s0, d.graph, d.sorting, done=set(), running={"pick_obj_01"})

    assert r.diff.moved == ("obj_02",)
    assert set(r.diff.removed) == {"obj_01", "obj_03"}  # obj_01 is held: not table work
    assert r.diff.added == ("obj_05",)
    assert set(r.dropped) == {"pick_obj_03", "place_obj_03"}
    assert r.updated == ["pick_obj_02"]
    assert r.graph.tasks["pick_obj_02"].args["pick_xyz"] == pytest.approx([0.64, 0.0, 0.02])
    assert {"pick_obj_01", "place_obj_01"} <= set(r.graph.tasks), "running work is untouched"
    assert set(r.added) == {"pick_obj_05", "place_obj_05"}
    assert r.graph.tasks["place_obj_05"].args["target"] == "bin_1"
    assert r.sorting.assignment["obj_05"] == "red" and set(r.sorting.categories) == {"red", "blue"}
    assert "already exist" in vlm.calls[1] and "obj_05" in vlm.calls[1]
    assert "obj_04" not in vlm.calls[1], "the held object is not sent to the VLM"


def test_rescan_resorts_a_dropped_object() -> None:
    s0 = build_scene([det(0.2, 0.0)], IDENTITY)
    vlm = FakeVLM([answer({"red": "bin_1"}, {"obj_01": "red"}), answer({}, {"obj_01": "red"})])
    dec = Decomposer(vlm, BINS, annotate=False)
    d = dec.decompose(None, s0)
    # placed, then it fell out next to where it started
    s1 = build_scene([det(0.21, 0.0)], IDENTITY, s0)
    r = rescan(dec, None, s1, s0, d.graph, d.sorting, done=set(d.graph.tasks), running=set())
    assert set(r.added) == {"pick_obj_01_r1", "place_obj_01_r1"}


def test_rescan_without_change_is_quiet() -> None:
    s0 = two_objects()
    vlm = FakeVLM([GOOD])
    dec = Decomposer(vlm, BINS, annotate=False)
    d = dec.decompose(None, s0)
    s1 = build_scene([det(0.201, 0.0), det(0.6, 0.001)], IDENTITY, s0)
    r = rescan(dec, None, s1, s0, d.graph, d.sorting, done=set(), running=set())
    assert not r.changed and not r.diff.changed and len(vlm.calls) == 1


# --- VLM safeguards (no GPU needed) ---------------------------------------------------------


def test_memory_plan_on_an_8gb_laptop_gpu() -> None:
    from hfteama.perception.vlm import VLMError, memory_plan

    # RTX 4060 Laptop: 8 GB, ~7.8 free. Qwen3-VL-4B does not fit in bf16 -> 4-bit.
    assert memory_plan(7.8, 8.0, 4.4, "auto", 0.85) == "4bit"
    assert memory_plan(22.0, 24.0, 4.4, "auto", 0.85) == "bf16"
    with pytest.raises(VLMError, match="not enough GPU memory"):
        memory_plan(7.8, 8.0, 4.4, "none", 0.85)
    with pytest.raises(VLMError):
        memory_plan(2.0, 8.0, 4.4, "auto", 0.85)  # another program holds the GPU


def test_vlm_failure_is_safe() -> None:
    from hfteama.perception import VLMError

    r = StepChecker(FakeVLM([VLMError("GPU at 90 C")]), BINS).check(_place_task(), None)
    assert not r.done and "GPU at 90 C" in r.reason
    with pytest.raises(DecompositionError, match="VLM unavailable"):
        Decomposer(FakeVLM([VLMError("oom")]), BINS, annotate=False).decompose(None, two_objects())
