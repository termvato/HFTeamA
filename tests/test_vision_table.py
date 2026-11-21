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
