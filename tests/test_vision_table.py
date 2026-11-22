"""Overhead-camera geometry and scene detection (synthetic images, no camera)."""

import pytest

np = pytest.importorskip("numpy")

from hfteama.vision.detect import Box, _hull, box_footprint  # noqa: E402
from hfteama.vision.table import (  # noqa: E402
    CameraPose,
    Intrinsics,
    centre_pixel,
    pixel_to_table,
    project,
)

INTR = Intrinsics(
    K=np.array([[1416.4, 0, 991.5], [0, 1415.2, 490.8], [0, 0, 1.0]]),
    D=np.array([0.029, -0.151, -0.0068, 0.0013, 0.087]),
)
TAGS = {0: (0.0, 0.0), 3: (0.73, 0.0), 2: (0.0, 0.335), 1: (0.73, 0.335)}


def oblique_pose():
    """Camera 0.71 m up, in front of the table, looking down and forward (like the real one)."""
    eye = np.array([0.32, -0.32, 0.71])
    look = np.array([0.365, 0.2, 0.0])
    z = (look - eye) / np.linalg.norm(look - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return CameraPose(R=R, t=-R @ eye)


def test_pixel_to_table_inverts_project() -> None:
    pose = oblique_pose()
    pts = np.array([[0.1, 0.05, 0.0], [0.6, 0.3, 0.0], [0.37, 0.03, 0.08]])
    px = project(pts, pose, INTR)
    for p, z in zip(pts, (0.0, 0.0, 0.08), strict=True):
        back = pixel_to_table(project([p], pose, INTR), pose, INTR, z)[0]
        assert back == pytest.approx(p, abs=1e-6)
    assert px.shape == (3, 2)


def test_centre_pixel_is_diagonal_intersection() -> None:
    corners = np.array([[0, 0], [10, 0], [12, 8], [-2, 8]], float)
    c = centre_pixel(corners)
    # on both diagonals
    for a, b in ((0, 2), (1, 3)):
        d = corners[b] - corners[a]
        e = c - corners[a]
        assert abs(d[0] * e[1] - d[1] * e[0]) < 1e-9


def test_box_footprint_covers_base_and_rim_shadow() -> None:
    cam = np.array([0.3, -0.3, 0.7])
    box = Box(x=0.37, y=0.1, radius=0.09)
    hull = box_footprint(box, 0.08, cam)
    xs, ys = hull[:, 0], hull[:, 1]
    # the rim's shadow lies beyond the base, away from the camera
    assert ys.max() > box.y + box.radius + 0.01
    assert ys.min() == pytest.approx(box.y - box.radius, abs=2e-3)
    assert xs.min() < box.x - box.radius + 1e-3


def test_hull_of_square_with_inner_point() -> None:
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
    assert len(_hull(pts)) == 4


# --- synthetic scene, needs OpenCV ---------------------------------------------------------


def render_scene(pose, size=(1920, 1080), tag_side=0.07):
    """Draw a wood-coloured table with the 4 tags, two pieces and a flat ring (the 'box' with
    height 0) as seen by `pose`. Returns the BGR image."""
    cv2 = pytest.importorskip("cv2")
    mpp = 0.001
    x0, x1, y0, y1 = -0.1, 0.83, -0.1, 0.45
    W, H = int((x1 - x0) / mpp), int((y1 - y0) / mpp)
    tex = np.full((H, W, 3), (80, 150, 200), np.uint8)  # BGR wood

    def px(x, y):
        return int(round((x - x0) / mpp)), int(round((y1 - y) / mpp))

    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    n = int(tag_side / mpp)
    for i, (tx, ty) in TAGS.items():
        m = cv2.aruco.generateImageMarker(d, i, n)
        u, v = px(tx, ty)
        q = n // 2 + 15
        tex[v - q : v + q, u - q : u + q] = 255  # white margin
        tex[v - n // 2 : v - n // 2 + n, u - n // 2 : u - n // 2 + n] = m[..., None]
    cv2.rectangle(tex, px(0.20, 0.20), px(0.27, 0.22), (200, 60, 30), -1)  # blue stick, along x
    cv2.circle(tex, px(0.50, 0.25), 20, (240, 240, 240), -1)  # white disc, 4 cm
    cv2.circle(tex, px(0.37, 0.08), 90, (60, 110, 160), 6)  # box rim, 18 cm
    # table plane (z = 0) -> image homography from 4 texture corners
    src = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    world = np.array([[x0, y1, 0], [x1, y1, 0], [x1, y0, 0], [x0, y0, 0]])
    dst = project(world, pose, Intrinsics(INTR.K, np.zeros(5))).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(tex, Hm, size, borderValue=(30, 30, 30))


def test_detect_scene_on_synthetic_image() -> None:
    pytest.importorskip("cv2")
    from hfteama.vision.detect import detect_scene
    from hfteama.vision.table import detect_tags, fit_pose

    truth = oblique_pose()
    img = render_scene(truth)
    intr = Intrinsics(INTR.K, np.zeros(5))  # rendered without lens distortion
    tags = detect_tags(img)
    assert set(tags) == set(TAGS)
    pose = fit_pose({i: centre_pixel(c) for i, c in tags.items()}, TAGS, intr)
    assert pose.rms_px < 1.0
    assert pose.position == pytest.approx(truth.position, abs=0.01)

    cfg = {
        "table": {"tags": {str(k): v for k, v in TAGS.items()}, "tag_mask": 0.11},
        "box": {"diameter": 0.18, "height": 0.0},
        "pieces": {"grasp_z": 0.0, "min_area_cm2": 1.0},
        "arms": {},
    }
    scene = detect_scene(img, pose, intr, cfg)
    assert scene.box is not None
    assert (scene.box.x, scene.box.y) == pytest.approx((0.37, 0.08), abs=0.006)
    found = sorted(scene.pieces, key=lambda p: p.x)
    assert len(found) == 2
    stick, disc = found
    assert (stick.x, stick.y) == pytest.approx((0.235, 0.21), abs=0.005)
    assert stick.length == pytest.approx(0.07, abs=0.006)
    assert stick.width == pytest.approx(0.02, abs=0.005)
    # the jaws close across the stick: along table y
    assert abs(np.sin(stick.yaw)) == pytest.approx(1.0, abs=0.02)
    assert (disc.x, disc.y) == pytest.approx((0.50, 0.25), abs=0.005)


def test_body_footprint_reaches_the_shadow() -> None:
    from hfteama.vision.detect import body_footprint

    cam = np.array([0.3, -0.3, 0.7])
    # a vertical post 20 cm high at (0.3, 0.3): its shadow points away from the camera
    polys = body_footprint([[0.3, 0.3, 0.0], [0.3, 0.3, 0.2]], 0.02, cam)
    ys = np.concatenate([p[:, 1] for p in polys])
    shadow_y = -0.3 + (0.3 + 0.3) * 0.7 / 0.5
    assert ys.max() == pytest.approx(shadow_y + 0.02, abs=2e-3)
    assert ys.min() == pytest.approx(0.28, abs=2e-3)


def test_touching_blobs_are_split() -> None:
    cv2 = pytest.importorskip("cv2")
    from hfteama.vision.detect import _split_touching

    mask = np.zeros((200, 300), np.uint8)
    cv2.circle(mask, (80, 100), 40, 255, -1)
    cv2.circle(mask, (190, 100), 40, 255, -1)
    cv2.rectangle(mask, (110, 95), (160, 105), 255, -1)  # thin bridge
    assert len(_split_touching(mask)) == 2
    single = np.zeros((200, 300), np.uint8)
    cv2.rectangle(single, (50, 80), (250, 110), 255, -1)
    assert len(_split_touching(single)) == 1


def test_dark_and_green_regions_are_ignored_but_small_spots_kept() -> None:
    cv2 = pytest.importorskip("cv2")
    from hfteama.vision.detect import _ignored

    # top-down image: row 0 = front (camera side), last row = back (where the arms stand)
    top = np.full((300, 400, 3), (60, 140, 210), np.uint8)  # BGR wood
    cv2.rectangle(top, (10, 190), (110, 299), (20, 20, 20), -1)  # black arm, from the back
    cv2.rectangle(top, (250, 190), (350, 299), (150, 210, 140), -1)  # mint-green arm
    cv2.circle(top, (200, 60), 6, (25, 25, 25), -1)  # dark cap of a piece, ~1 cm2
    cv2.circle(top, (100, 60), 15, (150, 60, 20), -1)  # blue lip balm
    cv2.rectangle(top, (150, 40), (290, 90), (90, 25, 10), -1)  # big dark-blue piece
    cv2.rectangle(top, (300, 20), (380, 100), (20, 20, 20), -1)  # a box's black print, 64 cm2
    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.float32)
    m = _ignored(top, lab, {"grow": 0.01, "min_area_cm2": 40.0})
    assert m[250, 60] and m[250, 300]
    assert m[250, 118]  # grown by 1 cm past the arm's edge
    assert not m[60, 200] and not m[60, 100]
    assert not m[65, 220]  # dark but blue: a piece, not an arm
    assert not m[60, 340]  # black, big, but in the middle of the table: a piece's print


def test_fit_box_finds_the_base_centre_from_the_silhouette() -> None:
    pytest.importorskip("cv2")
    from hfteama.vision.detect import box_silhouette, fit_box

    cam = np.array([0.367, 0.637, 0.726])
    dims = (0.058, 0.058, 0.039)
    truth = (0.20, 0.15, np.radians(12))
    outline = box_silhouette(*truth, dims, cam)
    # start from the blob's centre (pushed away from the camera by the top) and a rough yaw
    x, y, yaw = fit_box(outline, dims, cam, outline.mean(0), np.radians(5))
    assert (x, y) == pytest.approx(truth[:2], abs=0.003)
    assert np.hypot(*outline.mean(0) - truth[:2]) > 0.005  # the plain blob centre is off


def _tag_corners(pose, x, y, z, side, yaw):
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-u[1], u[0]])
    pts = [
        np.array([x, y]) + a * side / 2 * u + b * side / 2 * v
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]
    return project([[*p, z] for p in pts], pose, INTR)


def test_box_tag_gives_the_box_centre_and_a_loose_tag_is_ignored() -> None:
    from hfteama.vision.detect import pieces_from_tags

    pose = oblique_pose()
    pc = {
        "tags": {"size": 0.029, "size_tol": 0.04},
        "object_height": 0.039,
        "object_width": 0.058,
        "object_length": 0.058,
    }
    on_box = _tag_corners(pose, 0.25, 0.12, 0.039, 0.029, np.radians(20))
    loose = _tag_corners(pose, 0.45, 0.20, 0.0, 0.029, 0.0)  # lying on the table
    found = pieces_from_tags([(5, on_box), (5, loose)], pose, INTR, pc, (-0.1, 0.9, -0.1, 0.5))
    assert len(found) == 1
    assert (found[0].x, found[0].y) == pytest.approx((0.25, 0.12), abs=1e-4)
    assert np.degrees(found[0].yaw) == pytest.approx(20, abs=0.5)


def test_small_tag_with_a_reference_id_is_not_the_reference() -> None:
    from hfteama.vision.table import split_reference

    big = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    small = big * 0.35 + 300
    refs, others = split_reference([(0, small), (0, big), (3, big + 500)], {0, 1, 2, 3})
    assert refs[0] is big and set(refs) == {0, 3}
    assert len(others) == 1 and others[0][1] is small
