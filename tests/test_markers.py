"""Camera calibration from ArUco markers, on a rendered image with a known camera."""

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hfteama.perception.markers import calibrate_from_markers, detect_markers  # noqa: E402

W, H, F = 1920, 1080, 1400.0
S = 0.07
MARKERS = {0: (0.0, 0.0, 0.0), 2: (-0.10, 0.37, 1.0), 3: (0.62, 0.02, 178.0)}  # x, y, yaw deg


def camera() -> tuple[np.ndarray, np.ndarray]:
    """Camera ~0.75 m above, in front of the table, looking down and forwards (like the C920)."""
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    eye, target = np.array([0.31, -0.25, 0.75]), np.array([0.30, 0.25, 0.0])
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # table -> camera rotation (rows: camera axes in table frame)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ eye
    return K, T  # T = T_cam_table


def render() -> np.ndarray:
    K, T = camera()
    img = np.full((H, W, 3), (150, 180, 210), np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    n = 240
    for mid, (x, y, yaw) in MARKERS.items():
        tag = cv2.cvtColor(cv2.aruco.generateImageMarker(d, mid, n), cv2.COLOR_GRAY2BGR)
        q = 0.3 * S  # white quiet zone around the black square
        big = cv2.copyMakeBorder(tag, 72, 72, 72, 72, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        half = S / 2 + q
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        corners = []
        for px, py in ((-half, half), (half, half), (half, -half), (-half, -half)):
            wx, wy = x + c * px - s * py, y + s * px + c * py
            p = K @ (T @ np.array([wx, wy, 0.0, 1.0]))[:3]
            corners.append(p[:2] / p[2])
        m = big.shape[0]
        Hm = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [m, 0], [m, m], [0, m]]), np.float32(corners)
        )
        warped = cv2.warpPerspective(big, Hm, (W, H), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full((m, m), 255, np.uint8), Hm, (W, H)) > 0
        img[mask] = warped[mask]
    return img


def test_markers_give_camera_and_layout() -> None:
    img = render()
    assert sorted(detect_markers(img)) == [0, 2, 3]
    m = calibrate_from_markers(img, S)
    assert m.rms_px < 1.0
    assert m.focal_px == pytest.approx(F, rel=0.05)
    _, T = camera()
    assert m.calib.T_table_cam[:3, 3] == pytest.approx(np.linalg.inv(T)[:3, 3], abs=0.02)
    for mid, (x, y, yaw) in MARKERS.items():
        mx, my, myaw = m.markers[mid]
        assert math.dist((mx, my), (x, y)) < 0.01, mid
        assert abs((myaw - yaw + 180) % 360 - 180) < 2.0, mid
    # a pixel back to the table: project a known point and recover it
    K, T = camera()
    p = K @ (T @ np.array([0.25, 0.30, 0.0, 1.0]))[:3]
    back = m.calib.pixel_to_table(p[0] / p[2], p[1] / p[2], 0.0)
    assert math.dist(back[:2], (0.25, 0.30)) < 0.01


def test_known_focal_is_kept() -> None:
    m = calibrate_from_markers(render(), S, focal_px=F)
    assert m.focal_px == pytest.approx(F)


def test_no_markers() -> None:
    with pytest.raises(ValueError, match="no ArUco"):
        calibrate_from_markers(np.zeros((480, 640, 3), np.uint8), S)
