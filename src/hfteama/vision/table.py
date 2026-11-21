"""Table geometry seen by the overhead camera: camera pose from the reference tags, and
pixel <-> table-plane mapping.

Table frame: origin at the centre of tag 0, x towards tag 3, y towards tag 2 (away from the
camera), z up, metres. The camera pose is fitted to the tag CENTRES only, so it depends on
neither the tag size nor how each tag is turned on the table (same method as group-6's
KapplaBuild vision).

Projection and ray casting are plain numpy (OpenCV's pinhole + 5-coefficient Brown model), so
they are testable without OpenCV. Only the pose fit (solvePnP) and tag detection need cv2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Intrinsics:
    K: np.ndarray  # 3x3
    D: np.ndarray  # k1, k2, p1, p2, k3
    size: tuple[int, int] | None = None  # (width, height) the calibration is valid for

    @classmethod
    def load(cls, path: str | Path) -> Intrinsics:
        """Read an OpenCV FileStorage YAML with K, D (and optionally image_width/height)."""
        import cv2

        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise FileNotFoundError(path)
        K, D = fs.getNode("K").mat(), fs.getNode("D").mat()
        w, h = fs.getNode("image_width"), fs.getNode("image_height")
        size = None if w.empty() else (int(w.real()), int(h.real()))
        fs.release()
        return cls(K=np.asarray(K, float), D=np.asarray(D, float).ravel(), size=size)


@dataclass
class CameraPose:
    """X_cam = R @ X_table + t."""

    R: np.ndarray
    t: np.ndarray
    rms_px: float = 0.0
    tags_used: tuple[int, ...] = ()

    @property
    def position(self) -> np.ndarray:
        """Camera centre in the table frame."""
        return -self.R.T @ self.t


def _distort(xn: np.ndarray, D: np.ndarray) -> np.ndarray:
    k1, k2, p1, p2, k3 = (list(D) + [0.0] * 5)[:5]
    x, y = xn[..., 0], xn[..., 1]
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], -1)


def project(points, pose: CameraPose, intr: Intrinsics) -> np.ndarray:
    """Table-frame points (N, 3) -> pixels (N, 2)."""
    P = np.atleast_2d(np.asarray(points, float)) @ pose.R.T + pose.t
    xn = P[:, :2] / P[:, 2:3]
    xd = _distort(xn, intr.D)
    K = intr.K
    return np.stack([K[0, 0] * xd[:, 0] + K[0, 2], K[1, 1] * xd[:, 1] + K[1, 2]], -1)


def undistort(pixels, intr: Intrinsics, iters: int = 20) -> np.ndarray:
    """Pixels (N, 2) -> normalised, undistorted image coordinates (N, 2)."""
    px = np.atleast_2d(np.asarray(pixels, float))
    K = intr.K
    xd = np.stack([(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1]], -1)
    xn = xd.copy()
    for _ in range(iters):  # fixed point: xn = xd - (distort(xn) - xn)
        xn = xd - (_distort(xn, intr.D) - xn)
    return xn


def pixel_to_table(pixels, pose: CameraPose, intr: Intrinsics, z: float = 0.0) -> np.ndarray:
    """Where the camera rays through `pixels` (N, 2) meet the plane at height z: (N, 3)."""
    xn = undistort(pixels, intr)
    rays = np.concatenate([xn, np.ones((len(xn), 1))], 1) @ pose.R  # = (R.T @ ray_cam).T
    o = pose.position
    lam = (z - o[2]) / rays[:, 2]
    return o + lam[:, None] * rays


def centre_pixel(corners) -> np.ndarray:
    """Tag centre as the intersection of its diagonals (exact under perspective)."""
    p1, p2, p3, p4 = np.asarray(corners, float).reshape(4, 2)
    d1, d2 = p3 - p1, p4 - p2
    s = np.linalg.solve(np.array([d1, -d2]).T, p2 - p1)[0]
    return p1 + s * d1


def fit_pose(
    centres: dict[int, np.ndarray], tags: dict[int, tuple[float, float]], intr: Intrinsics
) -> CameraPose | None:
    """Camera pose from the pixel centres of the reference tags (at least 3 visible).

    With 3 tags the fit is exact (rms 0) and unchecked; with 4 the rms is a real check of
    the layout and the calibration.
    """
    import cv2

    ids = sorted(i for i in tags if i in centres)
    if len(ids) < 3:
        return None
    obj = np.array([[*tags[i], 0.0] for i in ids], np.float64)
    img = np.array([centres[i] for i in ids], np.float64)
    flags = cv2.SOLVEPNP_IPPE if len(ids) >= 4 else cv2.SOLVEPNP_SQPNP
    ok, rvec, tvec = cv2.solvePnP(obj, img, intr.K, intr.D, flags=flags)
    if not ok:
        return None
    if len(ids) >= 4:
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, intr.K, intr.D, rvec, tvec)
    pose = CameraPose(R=cv2.Rodrigues(rvec)[0], t=tvec.ravel(), tags_used=tuple(ids))
    if pose.position[2] <= 0:  # mirror solution below the table
        return None
    err = project(obj, pose, intr) - img
    pose.rms_px = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    return pose


def detect_all_tags(image, dictionary: str = "DICT_4X4_50") -> list[tuple[int, np.ndarray]]:
    """Every tag found, as (id, 4x2 corners); the same id may appear several times."""
    import cv2

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.minMarkerPerimeterRate = 0.01  # small tags (3 cm boxes) too
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)), params
    )
    corners, ids, _ = det.detectMarkers(image)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for c, i in zip(corners, ids.ravel(), strict=True)]


def detect_tags(image, dictionary: str = "DICT_4X4_50") -> dict[int, np.ndarray]:
    """{id: 4x2 corners} of every tag found; for a repeated id the largest wins."""
    out: dict[int, np.ndarray] = {}
    for i, c in detect_all_tags(image, dictionary):
        if i not in out or _side(c) > _side(out[i]):
            out[i] = c
    return out


def split_reference(dets, reference_ids, max_ratio: float = 0.6):
    """({id: corners} of the reference tags, [(id, corners)] of the rest). A reference id's
    largest detection is the reference; any detection less than max_ratio the size of the
    smallest reference (a small tag on a piece, whatever its id) goes to the rest."""
    refs: dict[int, np.ndarray] = {}
    for i, c in dets:
        if i in reference_ids and (i not in refs or _side(c) > _side(refs[i])):
            refs[i] = c
    if refs:  # a "reference" much smaller than the biggest is a piece tag sharing its id
        big = max(_side(c) for c in refs.values())
        refs = {i: c for i, c in refs.items() if _side(c) >= max_ratio * big}
    others = [(i, c) for i, c in dets if not (i in refs and c is refs[i])]
    if refs:
        small = min(_side(c) for c in refs.values())
        others = [(i, c) for i, c in others if _side(c) < max_ratio * small]
    return refs, others


def _side(c) -> float:
    return float(np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)))


def table_grid(x_range, y_range, mpp: float):
    """Table xy of every pixel of a top-down image: arrays X, Y of shape (rows, cols), row 0
    at the far edge (max y) so the image looks like the camera view."""
    xs = np.arange(x_range[0], x_range[1], mpp)
    ys = np.arange(y_range[1], y_range[0], -mpp)
    return np.meshgrid(xs, ys)


def warp_to_plane(image, pose: CameraPose, intr: Intrinsics, X, Y, z: float = 0.0):
    """Resample `image` onto the plane z over the table grid (X, Y): a top-down view."""
    import cv2

    P = np.stack([X, Y, np.full_like(X, z)], -1).reshape(-1, 3)
    uv = project(P, pose, intr).reshape(X.shape + (2,)).astype(np.float32)
    return cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)
