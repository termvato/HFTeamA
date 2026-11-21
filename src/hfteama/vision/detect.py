"""Find the round box and the pieces on the table in one overhead frame.

Everything happens in top-down images resampled onto a table plane (table.warp_to_plane), at
MM_PER_PX resolution, so sizes and thresholds are in real units:

- Box: an open cylinder of known diameter and height. Its rim is an exact circle in the image
  resampled at the rim's height, found with a Hough transform for that radius.
- Pieces: everything inside the reference-tag rectangle whose colour differs from the table
  (the median colour of the area), minus the tags, the arm bases and the box (its base and its
  rim seen from the camera). Each blob gives a centre, size and orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hfteama.vision.table import (
    CameraPose,
    Intrinsics,
    centre_pixel,
    pixel_to_table,
    project,
    table_grid,
)

MM_PER_PX = 1.0


@dataclass
class Box:
    x: float
    y: float
    radius: float  # m, rim radius used for the search
    votes: float = 0.0  # Hough accumulator of the circle found


@dataclass
class Piece:
    x: float  # grasp point, table frame (m)
    y: float
    length: float  # m, long side of the blob's minimum-area rectangle
    width: float  # m, short side: what the jaws close on
    yaw: float  # rad, table frame: direction of the short side (the closing direction)
    area_cm2: float
    outline: np.ndarray = field(repr=False, default_factory=lambda: np.zeros((0, 2)))  # table xy
    tag_id: int | None = None  # the tag on its top, if found by tag


@dataclass
class Scene:
    box: Box | None
    pieces: list[Piece]
    mask: np.ndarray | None = field(repr=False, default=None)  # piece mask (top-down)
    top_down: np.ndarray | None = field(repr=False, default=None)  # top-down colour image
    extent: tuple[float, float, float, float] = (0, 0, 0, 0)  # x0, x1, y0, y1 of the images


def region_extent(tags: dict[int, tuple[float, float]]):
    xs, ys = [p[0] for p in tags.values()], [p[1] for p in tags.values()]
    return min(xs), max(xs), min(ys), max(ys)


def _warp(image, pose, intr, extent, z):
    from hfteama.vision.table import warp_to_plane

    X, Y = table_grid(extent[:2], extent[2:], MM_PER_PX / 1000)
    return warp_to_plane(image, pose, intr, X, Y, z), X, Y


def find_box(image, pose: CameraPose, intr: Intrinsics, extent, diameter, height) -> Box | None:
    import cv2

    r = diameter / 2
    m = r * 0.6  # the rim may stick out of the tag rectangle a little
    ext = (extent[0] - m, extent[1] + m, extent[2] - m, extent[3] + m)
    top, X, _ = _warp(image, pose, intr, ext, height)
    gray = cv2.GaussianBlur(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    rpx = r * 1000 / MM_PER_PX
    circles = cv2.HoughCirclesWithAccumulator(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=rpx,
        param1=80,
        param2=25,
        minRadius=int(rpx * 0.85),
        maxRadius=int(rpx * 1.15),
    )
    if circles is None:
        return None
    cx, cy, _, votes = circles[0][0]
    x = ext[0] + cx * MM_PER_PX / 1000
    y = ext[3] - cy * MM_PER_PX / 1000
    return Box(x=float(x), y=float(y), radius=r, votes=float(votes))


def box_footprint(box: Box, height: float, camera_position: np.ndarray, n: int = 48):
    """Table xy outline (z = 0) that the box hides from the camera: the hull of its base and of
    its rim projected along the camera rays onto the table."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    base = np.stack([box.x + box.radius * np.cos(a), box.y + box.radius * np.sin(a)], -1)
    o = np.asarray(camera_position, float)
    s = o[2] / (o[2] - height)  # rim point -> its shadow on z = 0, scaled from the camera
    rim = o[:2] + (base - o[:2]) * s
    pts = np.concatenate([base, rim])
    return _hull(pts)


def _hull(pts: np.ndarray) -> np.ndarray:
    """Convex hull (monotone chain), counter-clockwise."""
    pts = np.unique(np.round(pts, 9), axis=0)
    if len(pts) < 3:
        return pts

    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and _cross(h[-1] - h[-2], p - h[-2]) <= 0:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def body_footprint(points, radius: float, camera_position, n: int = 12) -> list[np.ndarray]:
    """Table outlines (z = 0) hidden by a body made of capsules along `points` (table frame,
    (N, 3), z >= 0): for each segment, the hull of the capsule's ends straight below and of
    their shadows along the camera rays."""
    o = np.asarray(camera_position, float)
    pts = np.asarray(points, float)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = radius * np.stack([np.cos(a), np.sin(a)], -1)
    out = []
    for p, q in zip(pts[:-1], pts[1:], strict=True):
        ends = []
        for e in (p, q):
            z = min(max(e[2], 0.0), o[2] * 0.9)
            shadow = o[:2] + (e[:2] - o[:2]) * o[2] / (o[2] - z)
            ends += [e[:2] + ring, shadow + ring]
        out.append(_hull(np.concatenate(ends)))
    return out


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _to_px(xy, extent):
    xy = np.asarray(xy, float)
    return np.stack(
        [(xy[..., 0] - extent[0]) * 1000 / MM_PER_PX, (extent[3] - xy[..., 1]) * 1000 / MM_PER_PX],
        -1,
    )


def find_pieces(
    image,
    pose: CameraPose,
    intr: Intrinsics,
    extent,
    *,
    tags: dict[int, tuple[float, float]],
    tag_mask: float,
    exclude_discs: list[tuple[float, float, float]] = (),
    exclude_polygons: list[np.ndarray] = (),
    colour_threshold: float = 25.0,
    min_area_cm2: float = 1.0,
    max_area_cm2: float = 80.0,
    grasp_z: float = 0.015,
    ignore: dict | None = None,
    max_length: float = 0.15,
    box_dims: tuple[float, float, float] | None = None,
):
    """Pieces inside `extent` (x0, x1, y0, y1). Returns (pieces, mask, top_down image).

    box_dims: (length, width, height) m of the pieces when they are all the same box: each
    blob is then fitted with the silhouette such a box casts on the table seen from the
    camera (top and sides), which gives the centre of its base (fit_box).

    ignore: colours of things that are never pieces (the arms, their wrist cameras); see
    _ignored."""
    import cv2

    top, X, Y = _warp(image, pose, intr, extent, 0.0)
    valid = np.ones(X.shape, np.uint8)
    for tx, ty in tags.values():
        h = tag_mask / 2
        valid[(np.abs(X - tx) < h) & (np.abs(Y - ty) < h)] = 0
    for cx, cy, r in exclude_discs:
        valid[(X - cx) ** 2 + (Y - cy) ** 2 < r * r] = 0
    for poly in exclude_polygons:
        cv2.fillPoly(valid, [np.round(_to_px(poly, extent)).astype(np.int32)], 0)

    lab = cv2.cvtColor(top, cv2.COLOR_BGR2LAB).astype(np.float32)
    if ignore:
        valid[_ignored(top, lab, ignore) > 0] = 0
    table_colour = np.median(lab[valid > 0], axis=0)
    # lightness counts half: shadows and wood grain change it more than the hue
    dist = np.linalg.norm((lab - table_colour) * np.array([0.5, 1.0, 1.0], np.float32), axis=2)
    mask = ((dist > colour_threshold) & (valid > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    pieces = []
    contours = _split_touching(mask)
    px_area_cm2 = (MM_PER_PX / 10) ** 2
    for c in contours:
        area = cv2.contourArea(c) * px_area_cm2
        if not (min_area_cm2 <= area <= max_area_cm2):
            continue
        (u, v), (w, h), ang = cv2.minAreaRect(c)
        if max(w, h) * MM_PER_PX / 1000 > max_length:
            continue  # a cable or the table's edge, not a piece
        xy = _from_px(np.array([[u, v]]), extent)[0]
        # the blob is drawn on z = 0, but the camera sees the piece's top: re-cast the
        # centre at the grasp height so the grasp point sits over the piece
        if grasp_z:
            pix = project([[xy[0], xy[1], 0.0]], pose, intr)
            xy = pixel_to_table(pix, pose, intr, grasp_z)[0, :2]
        # minAreaRect angle: side w is at `ang` deg (image axes, y down); table y is up
        a = np.radians(ang)
        side_w = np.array([np.cos(a), -np.sin(a)])
        short = side_w if w <= h else np.array([-side_w[1], side_w[0]])
        yaw = float(np.arctan2(short[1], short[0]))
        length, width = max(w, h) * MM_PER_PX / 1000, min(w, h) * MM_PER_PX / 1000
        outline = _from_px(c.reshape(-1, 2).astype(float), extent)
        if box_dims is not None:
            bx, by, byaw = fit_box(outline, box_dims, pose.position, xy, yaw)
            xy, yaw = np.array([bx, by]), byaw
            length, width = max(box_dims[:2]), min(box_dims[:2])
        pieces.append(
            Piece(
                x=float(xy[0]),
                y=float(xy[1]),
                length=length,
                width=width,
                yaw=yaw,
                area_cm2=float(area),
                outline=outline,
            )
        )
    return pieces, mask, top


def box_silhouette(x, y, yaw, dims, camera_position) -> np.ndarray:
    """Table outline (z = 0) of what a box (length, width, height) standing at (x, y), turned
    by yaw (its width axis heading, rad), hides from the camera: the hull of its base and of
    its top corners cast along the camera rays onto the table."""
    L, W, H = dims
    u = np.array([np.cos(yaw), np.sin(yaw)])  # width axis
    v = np.array([-u[1], u[0]])  # length axis
    base = np.array([[x, y] + su * W / 2 * u + sv * L / 2 * v for su in (-1, 1) for sv in (-1, 1)])
    o = np.asarray(camera_position, float)
    top = o[:2] + (base - o[:2]) * o[2] / (o[2] - H)
    return _hull(np.concatenate([base, top]))


def fit_box(outline, dims, camera_position, xy0, yaw0, mm: float = 1.0):
    """(x, y, yaw) of the box base whose silhouette best overlaps `outline` (table xy, the
    blob's contour), searched around (xy0, yaw0). The silhouette is the box's base plus its
    top seen from the camera, so the result is the base centre, not the blob centre."""
    import cv2

    outline = np.asarray(outline, float)
    pad = 0.05
    x0, y0 = outline.min(0) - pad
    x1, y1 = outline.max(0) + pad
    shape = (int((y1 - y0) * 1000 / mm) + 1, int((x1 - x0) * 1000 / mm) + 1)

    def raster(poly):
        img = np.zeros(shape, np.uint8)
        px = np.round((np.asarray(poly) - [x0, y0]) * 1000 / mm).astype(np.int32)
        cv2.fillPoly(img, [px], 1)
        return img

    blob = raster(outline).astype(bool)

    def iou(x, y, yaw):
        sil = raster(box_silhouette(x, y, yaw, dims, camera_position)).astype(bool)
        return (blob & sil).sum() / max((blob | sil).sum(), 1)

    # start from the blob centre pulled back by the top's shift, then refine on a grid
    o = np.asarray(camera_position, float)
    k = o[2] / (o[2] - dims[2] / 2)
    best = (iou(*xy0, yaw0), *xy0, yaw0)
    start = o[:2] + (np.asarray(xy0) - o[:2]) / k
    best = max(best, (iou(*start, yaw0), *start, yaw0))
    for step, span in ((0.004, 0.03), (0.001, 0.005)):
        _, bx, by, byaw = best
        for dyaw in np.radians([-8, -4, 0, 4, 8]) if step > 0.002 else [0.0]:
            for dx in np.arange(-span, span + 1e-9, step):
                for dy in np.arange(-span, span + 1e-9, step):
                    cand = (iou(bx + dx, by + dy, byaw + dyaw), bx + dx, by + dy, byaw + dyaw)
                    if cand[0] > best[0]:
                        best = cand
    _, bx, by, byaw = best
    return float(bx), float(by), float(byaw)


def _ignored(top, lab, ig: dict):
    """Mask of large dark grey/black or green regions reaching the back edge (the black and
    the green arm, the wrist cameras, cables), grown by ig["grow"] m so their edges and servo
    wires go too.
    Small dark or green spots (a cap, a logo) stay: pieces are never removed for a detail."""
    import cv2

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    # dark AND colourless (the black arm, cameras, cables); a dark blue lip balm is not
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    dark = (lab[..., 0] < ig.get("dark_l_max", 75)) & (chroma < ig.get("dark_chroma_max", 20))
    h0, h1 = ig.get("green_hue", (30, 75))  # OpenCV hue, 0..180
    green = (hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= ig.get("green_s_min", 50))
    m = ((dark | green) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(m)
    min_px = ig.get("min_area_cm2", 15.0) / (MM_PER_PX / 10) ** 2
    # the arms come in from the back of the table (the image's last rows): a dark or green
    # region that doesn't reach that edge is a piece's print, not an arm
    band = int(ig.get("back_band", 0.03) * 1000 / MM_PER_PX)
    rows = m.shape[0]
    big = np.zeros_like(m)
    for i in range(1, n):
        reaches_back = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] >= rows - band
        if stats[i, cv2.CC_STAT_AREA] >= min_px and (reaches_back or not ig.get("from_back", True)):
            big[lab_cc == i] = 255
    g = max(1, int(ig.get("grow", 0.02) * 1000 / MM_PER_PX))
    return cv2.dilate(big, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))


def _split_touching(mask, solidity: float = 0.85):
    """Outer contours of the blobs, with blobs that look like several touching pieces (convex
    hull much bigger than the blob) split by a watershed on the distance to the background."""
    import cv2

    out = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.contourArea(cv2.convexHull(c))
        if area < 200 or hull == 0 or area / hull >= solidity:
            out.append(c)
            continue
        x, y, w, h = cv2.boundingRect(c)
        blob = np.zeros((h + 2, w + 2), np.uint8)
        cv2.drawContours(blob, [c - [x - 1, y - 1]], -1, 255, -1)
        dist = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
        n, markers = cv2.connectedComponents((dist > 0.5 * dist.max()).astype(np.uint8))
        if n <= 2:  # one core: a single (odd-shaped) piece
            out.append(c)
            continue
        markers = markers + 1
        markers[blob == 0] = 0
        markers[(blob > 0) & (dist <= 0.5 * dist.max())] = 0
        ws = cv2.watershed(cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
        for label in range(2, n + 1):
            part = ((ws == label) & (blob > 0)).astype(np.uint8)
            cs, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            out += [k + [x - 1, y - 1] for k in cs]
    return out


def _from_px(uv, extent):
    uv = np.asarray(uv, float)
    return np.stack(
        [extent[0] + uv[:, 0] * MM_PER_PX / 1000, extent[3] - uv[:, 1] * MM_PER_PX / 1000], -1
    )


def pieces_from_tags(dets, pose: CameraPose, intr: Intrinsics, pc: dict, extent) -> list[Piece]:
    """One piece per small tag lying on a box top: the tag centre cast onto the plane of the
    top (object_height) is the box centre; the tag's edge gives its turn. A tag whose side
    on that plane is not about tag_size (tolerance tag_size_tol) is not on a box top."""
    t = pc["tags"]
    size, tol = t.get("size", 0.03), t.get("size_tol", 0.2)
    H = pc["object_height"]
    L, W = pc.get("object_length", pc["object_width"]), pc["object_width"]
    x0, x1, y0, y1 = extent
    out = []
    for tag_id, c in dets:
        # centre = crossing of the diagonals (exact under perspective, unlike the mean)
        P = pixel_to_table(np.vstack([c, [centre_pixel(c)]]), pose, intr, H)[:, :2]
        corners, centre = P[:4], P[4]
        side = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))
        if abs(side - size) > tol * size or not (x0 <= centre[0] <= x1 and y0 <= centre[1] <= y1):
            continue
        e = corners[1] - corners[0]
        yaw = float(np.arctan2(e[1], e[0]))
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-u[1], u[0]])
        base = [
            centre + su * W / 2 * u + sv * L / 2 * v
            for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        out.append(
            Piece(
                x=float(centre[0]),
                y=float(centre[1]),
                length=max(L, W),
                width=min(L, W),
                yaw=yaw,
                area_cm2=L * W * 1e4,
                outline=np.array(base),
                tag_id=tag_id,
            )
        )
    return out


def detect_scene(
    image, pose: CameraPose, intr: Intrinsics, cfg: dict, arm_chains=(), piece_tags=None
) -> Scene:
    """Box and pieces, with the masks from the sorter config (configs/sorter.json).

    arm_chains: the arms' current polylines in the table frame (from FK), masked out with
    their shadows; without them only a disc around each base is masked.
    piece_tags: [(id, corners)] of the small tags seen (not the reference ones); with
    pieces.tags.enabled the pieces come from them (pieces_from_tags) instead of colour."""
    tags = {int(k): tuple(v) for k, v in cfg["table"]["tags"].items()}
    bc, pc = cfg["box"], cfg["pieces"]
    # the area searched for pieces: the tag rectangle grown by region_margin on every side
    m = pc.get("region_margin", 0.0)
    x0, x1, y0, y1 = region_extent(tags)
    extent = (x0 - m, x1 + m, y0 - m, y1 + m)
    box = find_box(image, pose, intr, extent, bc["diameter"], bc["height"])
    discs = [(*a["base"], a.get("mask_radius", 0.12)) for a in cfg["arms"].values()]
    polys = []
    if box is not None:
        grown = Box(box.x, box.y, box.radius + pc.get("box_margin", 0.02))
        polys.append(box_footprint(grown, bc["height"], pose.position))
    for chain in arm_chains:
        polys += body_footprint(chain, pc.get("arm_mask_radius", 0.04), pose.position)
    pieces, mask, top = find_pieces(
        image,
        pose,
        intr,
        extent,
        tags=tags,
        tag_mask=cfg["table"].get("tag_mask", 0.11),
        exclude_discs=discs,
        exclude_polygons=polys,
        colour_threshold=pc.get("colour_threshold", 25.0),
        min_area_cm2=pc.get("min_area_cm2", 1.0),
        max_area_cm2=pc.get("max_area_cm2", 80.0),
        grasp_z=pc.get("centre_z", pc.get("grasp_z", 0.015)),
        ignore=pc.get("ignore"),
        max_length=pc.get("max_length", 0.15),
        box_dims=(
            (pc["object_length"], pc["object_width"], pc["object_height"])
            if pc.get("fit_box") and "object_height" in pc
            else None
        ),
    )
    if pc.get("tags", {}).get("enabled") and piece_tags is not None:
        pieces = pieces_from_tags(piece_tags, pose, intr, pc, extent)
    return Scene(box=box, pieces=pieces, mask=mask, top_down=top, extent=extent)
