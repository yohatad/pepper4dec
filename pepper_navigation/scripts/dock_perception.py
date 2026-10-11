#!/usr/bin/env python3
"""Find Pepper's charging station near its known map pose, from lidar and camera.

Pure numpy / OpenCV, no ROS, so the tests can drive every step with synthetic
data; dock_detector.py wraps it in a node.

The station is modelled as a white box on the floor (StationModel). The robot
already knows roughly where it is (the dock pose in the map); this code measures
how far it has moved from there. Both sensors reduce to the same problem, a set
of floor-plane points that belong to the station, which fit_station() turns into
the station's centre and heading:

  * lidar: points in a height band just above the floor, near the expected pose,
    minus any spot that also has taller points (walls, furniture, legs);
  * bottom camera: white pixels on the uniformly coloured floor. The blob's
    outline is cast onto the floor plane for its near edge (where the white
    front face meets the floor) and onto the plane of the station's top for the
    rest, so a white wall right behind the station cannot drag the estimate.

Conventions: poses are Pose2D (x, y, yaw) in the fixed frame. A station's yaw
points from the room INTO the station, the way the robot faces while it looks at
it from the staging pose. Depth runs along the yaw, width across it.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class StationModel:
    """Size of the charging station's box, in metres."""

    depth: float = 0.37
    width: float = 0.39
    height: float = 0.075


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class FitLimits:
    """How far a measurement may stray before it is rejected."""

    max_shift: float = 0.30        # m, station centre from where the map says it is
    max_yaw_error: float = 0.35    # rad, heading from the map's
    size_tolerance: float = 0.06   # m, measured width against the model's
    min_visible_depth: float = 0.5  # fraction of the model's depth that must be in view
    min_points: int = 30


@dataclass(frozen=True)
class StationFit:
    """A measured station pose, with the extents it was measured from."""

    pose: Pose2D
    depth: float
    width: float
    points: int


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def to_frame(points_xy, pose):
    """Points in the frame of `pose`: u along its yaw, v to its left."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    d = points_xy - np.array([pose.x, pose.y])
    return np.stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c], axis=1)


def from_frame(u, v, pose):
    """The fixed-frame point at (u, v) in the frame of `pose`."""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return pose.x + u * c - v * s, pose.y + u * s + v * c


def floor_height(z, bin_size=0.01):
    """The most common height in a crop of the cloud: the floor, which dominates it."""
    if len(z) == 0:
        return None
    lo = float(np.min(z))
    bins = max(1, int(math.ceil((float(np.max(z)) - lo) / bin_size)) + 1)
    hist, edges = np.histogram(z, bins=bins, range=(lo, lo + bins * bin_size))
    i = int(np.argmax(hist))
    in_bin = z[(z >= edges[i]) & (z < edges[i + 1])]
    return float(np.median(in_bin))


def _search_reach(model, limits):
    return limits.max_shift + 0.5 * math.hypot(model.depth, model.width)


def lidar_station_points(points, prior, model=StationModel(), limits=FitLimits(),
                         min_height=0.02, height_margin=0.03, tall_margin=0.08, cell=0.03):
    """Floor-plane (x, y) of the lidar points that can belong to the station.

    Keeps points near the expected pose and between min_height and the station's
    height plus height_margin above the floor. Any grid cell (and its
    neighbours) that also holds a point more than tall_margin above the
    station's top is part of something taller, a wall, a chair or a leg, and is
    dropped. tall_margin is well beyond the lidar's noise, so stray returns off
    the station's own top cannot carve holes in it.

    Args:
        points: (N, 3) points in the fixed frame.
        prior: where the map says the station is.

    Returns:
        (M, 2) points; empty if there is no floor to measure from.
    """
    reach = _search_reach(model, limits)
    d = np.hypot(points[:, 0] - prior.x, points[:, 1] - prior.y)
    near = points[d <= reach + 0.3]        # keep a ring of floor for floor_height()
    if len(near) < 10:
        return np.empty((0, 2))
    floor_z = floor_height(near[:, 2])
    z = near[:, 2] - floor_z
    top = model.height + height_margin
    band = (z >= min_height) & (z <= top) & (np.hypot(near[:, 0] - prior.x,
                                                      near[:, 1] - prior.y) <= reach)
    tall = near[z > model.height + tall_margin, :2]
    if len(tall) and band.any():
        origin = near[:, :2].min(axis=0)
        tall_ij = np.floor((tall - origin) / cell).astype(int)
        band_ij = np.floor((near[band, :2] - origin) / cell).astype(int)
        shape = np.maximum(tall_ij.max(axis=0), band_ij.max(axis=0)) + 2
        grid = np.zeros((shape[1], shape[0]), np.uint8)
        grid[tall_ij[:, 1], tall_ij[:, 0]] = 1
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        blocked = grid[band_ij[:, 1], band_ij[:, 0]] > 0
        band[np.flatnonzero(band)[blocked]] = False
    return near[band, :2]


def largest_cluster(points_xy, prior, model=StationModel(), cell=0.02):
    """The connected group of points that covers the expected footprint most.

    Rasterises the points, splits them into 8-connected components and keeps
    the one with the most points inside the footprint the map expects (ties
    and empty overlaps fall back to the component nearest its centre).
    """
    if len(points_xy) == 0:
        return points_xy
    origin = points_xy.min(axis=0) - cell
    ij = np.floor((points_xy - origin) / cell).astype(int)
    shape = ij.max(axis=0) + 2
    grid = np.zeros((shape[1], shape[0]), np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 255
    count, labels = cv2.connectedComponents(grid, connectivity=8)
    point_labels = labels[ij[:, 1], ij[:, 0]]
    local = to_frame(points_xy, prior)
    inside = (np.abs(local[:, 0]) <= model.depth / 2) & (np.abs(local[:, 1]) <= model.width / 2)
    best, best_key = None, None
    for label in range(1, count):
        members = point_labels == label
        dist = np.min(np.hypot(local[members, 0], local[members, 1]))
        key = (int(np.sum(inside & members)), -dist)
        if best_key is None or key > best_key:
            best, best_key = label, key
    return points_xy[point_labels == best]


def _closest_yaw(angle, reference):
    """The angle among angle + k * 90 degrees nearest the reference."""
    candidates = [angle + k * math.pi / 2 for k in range(4)]
    return min(candidates, key=lambda a: abs(wrap_angle(a - reference)))


def _near_edge(points_xy, frame, band):
    """The points within band of the near end (low u) of points_xy in frame."""
    u = to_frame(points_xy, frame)[:, 0]
    return points_xy[u <= np.percentile(u, 1.0) + band]


def fit_station(points_xy, prior, model=StationModel(), limits=FitLimits(),
                near_candidates=None, near_band=0.04):
    """Measure the station's pose from its points, or None if they do not fit.

    The heading comes from a line through the station's near edge, refined a
    few times so that a station turned away from the map's heading still has
    its whole edge selected. The near edge is the densest and least ambiguous
    part of either sensor's view: the lidar's returns off the front face, or
    the camera's front-face/floor boundary. Edges are then robust percentiles
    in that heading's frame. The centre is placed half the model's depth beyond
    the near edge. The far edge is not trusted for that: the lidar sees the
    top at a grazing angle and reads it short, and for the camera a white wall
    behind the station merges with it. The visible depth only has to show that
    enough of the station is in view.

    Args:
        points_xy: (N, 2) station points in the fixed frame.
        prior: where the map says the station is.
        near_candidates: optional (N, 2) floor-plane positions of the same
            points, row for row (the camera's outline cast onto the floor as
            well as onto the station's top). The near edge is taken from them,
            and a point only counts towards the width if both of its positions
            lie within the station's depth, which leaves out where a wall
            beside the station meets the floor. Defaults to points_xy.

    Returns:
        StationFit, or None when there are too few points, too little of the
        station is in view, its width is not the model's, or it moved further
        than limits allow.
    """
    if len(points_xy) < limits.min_points:
        return None
    source = points_xy if near_candidates is None else near_candidates
    if near_candidates is None:
        (_, _), (_, _), angle = cv2.minAreaRect(points_xy.astype(np.float32))
        yaw = _closest_yaw(math.radians(angle), prior.yaw)
    else:
        yaw = prior.yaw
    for _ in range(4):
        near = _near_edge(source, Pose2D(prior.x, prior.y, yaw), near_band)
        if len(near) < 5:
            return None
        line = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        new_yaw = _closest_yaw(math.atan2(line[1], line[0]) + math.pi / 2, prior.yaw)
        converged = abs(wrap_angle(new_yaw - yaw)) < 1e-3
        yaw = new_yaw
        if converged:
            break
    frame = Pose2D(prior.x, prior.y, yaw)
    u_near = float(np.median(to_frame(_near_edge(source, frame, near_band), frame)[:, 0]))
    local = to_frame(points_xy, frame)
    depth = float(np.percentile(local[:, 0], 98.0)) - u_near
    if depth < limits.min_visible_depth * model.depth:
        return None                                    # mostly hidden: do not guess
    # Width from the station's middle stretch only: whatever adjoins its back
    # (a wall, the camera's white wall) cannot widen it there.
    gate = local if near_candidates is None else to_frame(near_candidates, frame)
    far_limit = u_near + model.depth - 0.03
    middle = local[(local[:, 0] >= u_near + 0.02) & (local[:, 0] <= far_limit)
                   & (gate[:, 0] <= far_limit)]
    if len(middle) < limits.min_points // 2:
        return None
    v_lo, v_hi = (float(v) for v in np.percentile(middle[:, 1], [2.0, 98.0]))
    width = v_hi - v_lo
    if abs(width - model.width) > limits.size_tolerance:
        return None
    x, y = from_frame(u_near + 0.5 * model.depth, 0.5 * (v_lo + v_hi), frame)
    pose = Pose2D(x, y, yaw)
    if math.hypot(x - prior.x, y - prior.y) > limits.max_shift:
        return None
    if abs(wrap_angle(yaw - prior.yaw)) > limits.max_yaw_error:
        return None
    return StationFit(pose, depth, width, len(points_xy))


def measure_with_lidar(points, prior, model=StationModel(), limits=FitLimits()):
    """Station pose from a fixed-frame lidar cloud (N, 3), or None."""
    candidates = lidar_station_points(points, prior, model, limits)
    return fit_station(largest_cluster(candidates, prior, model), prior, model, limits)


# --------------------------------------------------------------------------
# Bottom camera
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    """A pinhole camera with plumb-bob distortion, posed in the fixed frame.

    rotation maps optical-frame vectors (x right, y down, z forward) into the
    fixed frame; position is the optical centre in the fixed frame.
    """

    k: np.ndarray            # 3x3
    d: np.ndarray            # distortion coefficients
    rotation: np.ndarray     # 3x3
    position: np.ndarray     # 3

    def project(self, points_xyz):
        """Pixels (N, 2) of fixed-frame points; NaN for points behind the camera."""
        pc = (points_xyz - self.position) @ self.rotation
        behind = pc[:, 2] <= 1e-6
        rvec, _ = cv2.Rodrigues(np.eye(3))
        uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3).astype(np.float64), rvec,
                                  np.zeros(3), self.k, self.d)
        uv = uv.reshape(-1, 2)
        uv[behind] = np.nan
        return uv

    def cast(self, pixels, plane_z):
        """Fixed-frame (x, y) where the rays through `pixels` meet z = plane_z."""
        norm = cv2.undistortPoints(pixels.reshape(-1, 1, 2).astype(np.float64),
                                   self.k, self.d).reshape(-1, 2)
        rays = np.hstack([norm, np.ones((len(norm), 1))]) @ self.rotation.T
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (plane_z - self.position[2]) / rays[:, 2]
        hits = self.position + s[:, None] * rays
        hits[(s <= 0) | ~np.isfinite(s)] = np.nan
        return hits[:, :2]


@dataclass(frozen=True)
class WhiteThresholds:
    """What counts as the station's white against the floor (HSV, 0-255)."""

    max_saturation: int = 60
    min_brightness: int = 60        # only excludes near-black; Otsu adapts to the light
    min_contrast: int = 40       # white must be this much brighter than the floor


def footprint_corners(pose, model, margin=0.0, z=0.0):
    """The station's footprint corners (4, 3) at height z, grown by margin."""
    hd, hw = model.depth / 2 + margin, model.width / 2 + margin
    return np.array([[*from_frame(u, v, pose), z]
                     for u, v in ((-hd, -hw), (hd, -hw), (hd, hw), (-hd, hw))])


def search_mask(camera, prior, model, limits, shape):
    """Image region where the station can appear: the grown footprint, floor to top."""
    corners = np.vstack([footprint_corners(prior, model, limits.max_shift, z)
                         for z in (0.0, model.height)])
    uv = camera.project(corners)
    mask = np.zeros(shape[:2], np.uint8)
    if np.isnan(uv).any():
        return mask
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.fillPoly(mask, [hull], 255)
    return mask


def white_mask(bgr, region, thresholds=WhiteThresholds()):
    """Pixels inside region that are white: unsaturated and brighter than the floor.

    The region holds the single-coloured floor and, when it is in view, the
    white station (and maybe a white wall behind it): two brightness groups.
    Otsu's threshold splits them; the bright group counts only if it clears the
    floor by min_contrast, so a region of plain floor yields no white at all.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    inside = region > 0
    empty = np.zeros(region.shape, np.uint8)
    if not inside.any():
        return empty
    values = val[inside].reshape(-1, 1)
    split, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark, bright = values[values <= split], values[values > split]
    if len(bright) == 0 or float(bright.mean()) - float(dark.mean()) < thresholds.min_contrast:
        return empty
    level = max(split + 1, thresholds.min_brightness)
    mask = ((sat <= thresholds.max_saturation) & (val >= level) & inside).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def camera_station_points(bgr, camera, prior, model=StationModel(), limits=FitLimits(),
                          thresholds=WhiteThresholds(), floor_z=0.0):
    """The station's outline from the bottom camera, as floor-plane points.

    Returns:
        (top_xy, floor_xy, mask): the white blob's outline cast onto the plane
        of the station's top and onto the floor, and the blob mask (for
        debugging). The outline's near edge is the bottom of the white front
        face, so only its floor cast is where the station really is; the rest of
        the outline is the top's edges. Empty arrays when no white blob covers
        the expected footprint.
    """
    region = search_mask(camera, prior, model, limits, bgr.shape)
    mask = white_mask(bgr, region, thresholds)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return np.empty((0, 2)), np.empty((0, 2)), mask
    expected = np.zeros(mask.shape, np.uint8)
    uv = camera.project(footprint_corners(prior, model, 0.0, floor_z + model.height))
    if not np.isnan(uv).any():
        cv2.fillPoly(expected, [uv.astype(np.int32)], 255)
    overlap = [int(np.sum((labels == i) & (expected > 0))) for i in range(count)]
    overlap[0] = -1
    best = int(np.argmax(overlap)) if max(overlap) > 0 else 1 + int(np.argmax(stats[1:, 4]))
    blob = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    top = camera.cast(outline, floor_z + model.height)
    floor = camera.cast(outline, floor_z)
    ok = np.isfinite(top).all(axis=1) & np.isfinite(floor).all(axis=1)
    return top[ok], floor[ok], blob


def measure_with_camera(bgr, camera, prior, model=StationModel(), limits=FitLimits(),
                        thresholds=WhiteThresholds(), floor_z=0.0):
    """Station pose from one bottom-camera image, or None."""
    top, floor, _ = camera_station_points(bgr, camera, prior, model, limits, thresholds,
                                          floor_z)
    return fit_station(top, prior, model, limits, near_candidates=floor)


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Estimate:
    pose: Pose2D
    sigma_xy: float      # m, 1-sigma
    sigma_yaw: float     # rad, 1-sigma


def fuse(estimates, max_disagreement_xy=0.05, max_disagreement_yaw=0.10):
    """Inverse-variance mean of the estimates, or None if any two disagree.

    Two sensors that disagree by more than the tolerances mean at least one is
    looking at something else; picking one would be a guess, so nothing is
    returned and the caller does not publish.
    """
    if not estimates:
        return None
    for i, a in enumerate(estimates):
        for b in estimates[i + 1:]:
            if math.hypot(a.pose.x - b.pose.x, a.pose.y - b.pose.y) > max_disagreement_xy:
                return None
            if abs(wrap_angle(a.pose.yaw - b.pose.yaw)) > max_disagreement_yaw:
                return None
    wxy = np.array([1.0 / e.sigma_xy ** 2 for e in estimates])
    wyaw = np.array([1.0 / e.sigma_yaw ** 2 for e in estimates])
    x = float(np.sum(wxy * [e.pose.x for e in estimates]) / np.sum(wxy))
    y = float(np.sum(wxy * [e.pose.y for e in estimates]) / np.sum(wxy))
    ref = estimates[0].pose.yaw
    yaw = ref + float(np.sum(wyaw * [wrap_angle(e.pose.yaw - ref) for e in estimates])
                      / np.sum(wyaw))
    return Pose2D(x, y, wrap_angle(yaw))


# --------------------------------------------------------------------------
# Dock database geometry
# --------------------------------------------------------------------------

def station_from_dock(dock, station_offset):
    """The station's centre from the docked robot pose.

    The docking server's dock pose is where base_footprint ends up when docked,
    with the dock's yaw pointing into the station; the station's centre lies
    station_offset metres further along that yaw.
    """
    x, y = from_frame(station_offset, 0.0, dock)
    return Pose2D(x, y, dock.yaw)
