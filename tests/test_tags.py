"""Boxes from tags on their tops, on a rendered table with a known camera."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hfteama.perception import (  # noqa: E402
    Bin,
    TagBox,
    TagConfig,
    TagDetector,
    build_scene,
    tag_sorting,
)
from hfteama.perception.markers import calibrate_from_markers  # noqa: E402
from hfteama.perception.tags import tag_yaw_deg  # noqa: E402

W, H, F = 1920, 1080, 1400.0
TABLE = {0: (0.0, 0.0, 0.0), 2: (-0.10, 0.37, 1.0), 3: (0.62, 0.02, 178.0)}  # 4x4, 7 cm, z = 0
TABLE_SIZE = 0.07
BOX_H = 0.03
BOX_TAG = 0.04
BINS = [Bin("bowl", "the bowl", (0.317, 0.126, 0.10), radius_m=0.08)]


def camera() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    eye, target = np.array([0.31, -0.25, 0.75]), np.array([0.30, 0.25, 0.0])
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return K, T  # T = T_cam_table


def paste(img, dictionary, mid, x, y, yaw_deg, size, z) -> None:
    K, T = camera()
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    n = 240
    tag = cv2.cvtColor(cv2.aruco.generateImageMarker(d, mid, n), cv2.COLOR_GRAY2BGR)
    pad = 72
    big = cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    half = size / 2 * (n + 2 * pad) / n
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    corners = []
    for px, py in ((-half, half), (half, half), (half, -half), (-half, -half)):
        wx, wy = x + c * px - s * py, y + s * px + c * py
        p = K @ (T @ np.array([wx, wy, z, 1.0]))[:3]
        corners.append(p[:2] / p[2])
    m = big.shape[0]
    Hm = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [m, 0], [m, m], [0, m]]), np.float32(corners)
    )
    warped = cv2.warpPerspective(big, Hm, (W, H), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((m, m), 255, np.uint8), Hm, (W, H)) > 0
    img[mask] = warped[mask]


def render(boxes: dict[int, tuple[float, float, float]], box_dict="DICT_APRILTAG_36h11"):
    img = np.full((H, W, 3), (150, 180, 210), np.uint8)
    for mid, (x, y, yaw) in TABLE.items():
        paste(img, "DICT_4X4_50", mid, x, y, yaw, TABLE_SIZE, 0.0)
    for mid, (x, y, yaw) in boxes.items():
        paste(img, box_dict, mid, x, y, yaw, BOX_TAG, BOX_H)
    return img


BOXES = {5: (0.20, 0.22, 30.0), 9: (0.45, 0.30, -60.0)}


def calib(img, **kw):
    return calibrate_from_markers(img, TABLE_SIZE, dictionary="DICT_4X4_50", **kw).calib


def test_tags_give_box_position_yaw_and_id() -> None:
    img = render(BOXES)
    cfg = TagConfig(height_m=BOX_H, boxes={5: TagBox(label="small red box")})
    dets = TagDetector(calib(img), cfg).detect(img)
    assert sorted(d.key for d in dets) == ["tag_05", "tag_09"]
    scene = build_scene(dets, calib(img))
    for tid, (x, y, yaw) in BOXES.items():
        o = scene.objects[f"tag_{tid:02d}"]
        assert math.dist(o.xyz[:2], (x, y)) < 0.005, tid
        assert o.xyz[2] == pytest.approx(BOX_H / 2)
        assert abs((tag_yaw_deg(o.quat_wxyz) - yaw + 180) % 360 - 180) < 2.0, tid
    assert scene.objects["tag_05"].label == "small red box"
    assert scene.objects["tag_09"].label == "box with tag 9"


def test_ids_survive_a_big_move() -> None:
    img1, img2 = render({5: (0.20, 0.22, 0.0)}), render({5: (0.50, 0.10, 90.0)})
    cal = calib(img1)
    det = TagDetector(cal, TagConfig(height_m=BOX_H))
    s1 = build_scene(det.detect(img1), cal)
    s2 = build_scene(det.detect(img2), cal, previous=s1)
    assert s1.ids() == s2.ids() == ["tag_05"]


def test_only_listed_and_table_ids_are_skipped() -> None:
    # boxes in the table's own dictionary: calibration and detector must keep them apart
    img = render({7: (0.20, 0.22, 10.0), 8: (0.45, 0.30, 0.0)}, box_dict="DICT_4X4_50")
    m = calibrate_from_markers(img, TABLE_SIZE, only_ids=TABLE)
    assert sorted(m.markers) == [0, 2, 3]
    assert m.rms_px < 1.0
    cfg = TagConfig(dictionary="DICT_4X4_50", height_m=BOX_H, only_listed=True,
                    boxes={7: TagBox()})  # fmt: skip
    det = TagDetector(m.calib, cfg, ignore_ids=m.markers)
    assert [d.key for d in det.detect(img)] == ["tag_07"]
    assert det.dropped == [(8, "not listed in the config")]


def test_tag_sorting_plans_without_a_vlm() -> None:
    img = render({**BOXES, 11: (0.317, 0.126, 0.0)})  # tag 11 already in the bowl
    cfg = TagConfig(
        height_m=BOX_H,
        boxes={5: TagBox(bin="bowl"), 11: TagBox(bin="bowl")},  # 9 has no bin: left alone
    )
    cal = calib(img)
    scene = build_scene(TagDetector(cal, cfg).detect(img), cal)
    dec = tag_sorting(scene, BINS, cfg)
    assert sorted(dec.graph.tasks) == ["pick_tag_05", "place_tag_05"]
    assert dec.already_sorted == ["tag_11"]
    pick = dec.graph.tasks["pick_tag_05"].args
    assert math.dist(pick["pick_xyz"][:2], BOXES[5][:2]) < 0.005
    assert dec.graph.tasks["place_tag_05"].args["target"] == "bowl"


def test_unknown_bin_is_refused() -> None:
    cfg = TagConfig(boxes={5: TagBox(bin="nowhere")})
    with pytest.raises(ValueError, match="unknown bin"):
        tag_sorting(build_scene([], calib(render({}))), BINS, cfg)
