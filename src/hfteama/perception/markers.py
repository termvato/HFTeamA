"""Camera calibration from ArUco markers of known size lying on the table.

No checkerboard session needed: the markers already on the table (4x4 dictionary, 7 cm black
square) give the camera pose and, from their coplanarity, the focal length.

- Table frame: origin at the reference marker's centre, x and y along that marker's printed axes,
  z up out of the table (OpenCV's ArUco marker frame).
- Fitted jointly over every detected corner: camera pose, focal length (square pixels, principal
  point at the image centre, no lens distortion) and the in-plane pose (x, y, yaw) of every
  marker other than the reference. Levenberg-Marquardt on the reprojection error.
- Output: hfteama.perception.frames.Calibration (T_table_cam, K, image_size), so the VLM pointing
  detector turns pixels into table coordinates with it, plus the marker layout and the fit error.

Needs OpenCV (opencv-python >= 4.7, which includes cv2.aruco).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from hfteama.perception.frames import Calibration

C920_FOCAL_1080P = 1360.0  # px; Logitech C920, 78 deg diagonal field of view at 1920x1080


@dataclass
class MarkerCalibration:
    calib: Calibration
    focal_px: float
    rms_px: float  # reprojection error of the fit
    markers: dict[int, tuple[float, float, float]]  # id -> (x, y, yaw_deg) in the table frame
    reference_id: int

    def summary(self) -> str:
        lines = [
            f"focal {self.focal_px:.0f} px, fit error {self.rms_px:.2f} px, "
            f"camera at {np.round(self.calib.T_table_cam[:3, 3], 3).tolist()} m "
            f"(marker {self.reference_id} frame)"
        ]
        for i, (x, y, yaw) in sorted(self.markers.items()):
            lines.append(f"  marker {i}: ({x:+.3f}, {y:+.3f}) m, yaw {yaw:+.1f} deg")
        return "\n".join(lines)


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the machine
        raise ImportError('marker calibration needs OpenCV: pip install -e ".[camera]"') from e
    return cv2


def detect_markers(image: Any, dictionary: str = "DICT_4X4_50") -> dict[int, np.ndarray]:
    """Marker id -> 4x2 corner pixels (sub-pixel), in ArUco order (TL, TR, BR, BL)."""
    cv2 = _cv2()
    img = np.asarray(image)
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
    out = {}
    for i, c in zip(ids.flatten().tolist(), corners, strict=True):
        c = c.reshape(4, 1, 2).astype(np.float32)
        out[int(i)] = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), crit).reshape(4, 2)
    return out


def _square(size: float) -> np.ndarray:
    s = size / 2
    return np.array([[-s, s], [s, s], [s, -s], [-s, -s]], float)


def _rodrigues(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def _project(p: np.ndarray, others: list[int], ref_pts: np.ndarray, sq: np.ndarray, c: tuple):
    """All corners' pixels for parameters p = [rvec, tvec, log f, (x, y, yaw) per other marker]."""
    R, t, f = _rodrigues(p[:3]), p[3:6], math.exp(p[6])
    pts = [np.c_[ref_pts, np.zeros(4)]]
    for k in range(len(others)):
        x, y, yaw = p[7 + 3 * k : 10 + 3 * k]
        cy, sy = math.cos(yaw), math.sin(yaw)
        xy = sq @ np.array([[cy, sy], [-sy, cy]]) + (x, y)
        pts.append(np.c_[xy, np.zeros(4)])
    P = np.vstack(pts) @ R.T + t
    return np.c_[f * P[:, 0] / P[:, 2] + c[0], f * P[:, 1] / P[:, 2] + c[1]]


def calibrate_from_markers(
    image: Any,
    marker_size_m: float,
    reference_id: int | None = None,
    focal_px: float | None = None,
    dictionary: str = "DICT_4X4_50",
    iterations: int = 100,
    only_ids: Any = None,
) -> MarkerCalibration:
    """Camera pose (and focal length) from the markers in one image of the table.

    `only_ids`: the table markers' ids; other markers of the dictionary (e.g. tags on boxes,
    which sit above the table and move) are left out of the fit. None = every marker found.
    """
    cv2 = _cv2()
    img = np.asarray(image)
    h, w = img.shape[:2]
    found = detect_markers(img, dictionary)
    if only_ids is not None:
        keep = {int(i) for i in only_ids}
        found = {i: c for i, c in found.items() if i in keep}
    if not found:
        raise ValueError("no ArUco markers found")
    ref = min(found) if reference_id is None else reference_id
    if ref not in found:
        raise ValueError(f"reference marker {ref} not in view (found {sorted(found)})")
    others = sorted(i for i in found if i != ref)
    sq = _square(marker_size_m)
    obj = np.c_[sq, np.zeros(4)].astype(np.float32)
    c = (w / 2, h / 2)
    f0 = focal_px or C920_FOCAL_1080P * w / 1920

    # initial guess: each marker on its own (IPPE), others expressed in the reference frame
    def pnp(i: int, f: float) -> tuple[np.ndarray, np.ndarray]:
        K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
        _, rv, tv = cv2.solvePnP(obj, found[i], K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return cv2.Rodrigues(rv)[0], tv.ravel()

    R0, t0 = pnp(ref, f0)
    p = [*cv2.Rodrigues(R0)[0].ravel(), *t0, math.log(f0)]
    for i in others:
        Ri, ti = pnp(i, f0)
        q = R0.T @ (ti - t0)
        Rrel = R0.T @ Ri
        p += [q[0], q[1], math.atan2(Rrel[1, 0], Rrel[0, 0])]
    p = np.array(p, float)
    meas = np.vstack([found[ref], *[found[i] for i in others]])

    def resid(q: np.ndarray) -> np.ndarray:
        return (_project(q, others, sq, sq, c) - meas).ravel()

    # Levenberg-Marquardt, numerical Jacobian (13 parameters at 3 markers: tiny)
    fix_f = focal_px is not None
    lam = 1e-3
    r = resid(p)
    for _ in range(iterations):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            d = np.zeros_like(p)
            d[k] = 1e-6 * max(1.0, abs(p[k]))
            J[:, k] = (resid(p + d) - r) / d[k]
        if fix_f:
            J[:, 6] = 0.0
        A = J.T @ J
        g = J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
        r_new = resid(p + step)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-10:
                break
        else:
            lam *= 10

    R, t, f = _rodrigues(p[:3]), p[3:6], math.exp(p[6])
    T_cam_table = np.eye(4)
    T_cam_table[:3, :3], T_cam_table[:3, 3] = R, t
    K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1.0]])
    markers = {ref: (0.0, 0.0, 0.0)}
    for k, i in enumerate(others):
        x, y, yaw = p[7 + 3 * k : 10 + 3 * k]
        markers[i] = (float(x), float(y), math.degrees(yaw))
    return MarkerCalibration(
        calib=Calibration(np.linalg.inv(T_cam_table), K=K, image_size=(w, h)),
        focal_px=f,
        rms_px=float(np.sqrt(np.mean(r.reshape(-1, 2) ** 2) * 2)),
        markers=markers,
        reference_id=ref,
    )
