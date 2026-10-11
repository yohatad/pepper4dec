"""A tiny ray caster that plays the dock's surroundings for the docking tests.

The scene is a floor at z = 0, the charging station box, and optionally a wall,
extra boxes and vertical cylinders (legs). lidar_scan() shoots rays over the
L2's front hemisphere from where sensor_tf.yaml mounts it; render() shoots one
ray per pixel of a Camera from dock_perception. Both take the nearest hit, so
density, occlusion and perspective come out as the real sensors would see them.
"""

from dataclasses import dataclass, field
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from dock_perception import Camera, from_frame, Pose2D, StationModel  # noqa: E402

FLOOR, STATION_TOP, STATION_FRONT, STATION_SIDE, WALL, OTHER = range(6)

# Pepper geometry from naoqi_driver2's pepper.urdf and pepper_slam's sensor_tf.yaml.
LIDAR_IN_BASE = np.array([0.133, 0.0, 0.2582])
BOTTOM_CAMERA_K = np.array([[558.570339530768, 0.0, 308.885375457296],
                            [0.0, 556.122943034837, 247.600724811385],
                            [0.0, 0.0, 1.0]])
BOTTOM_CAMERA_D = np.array([-0.0648763971625288, 0.0612520196884308, 0.0038281538281731,
                            -0.00551104078371959, 0.0])
# Bottom camera at HeadPitch 0.2 rad: optical centre and axis in base_footprint.
BOTTOM_CAMERA_IN_BASE = np.array([0.066, 0.0, 0.82 + 0.212])
BOTTOM_CAMERA_AXIS = np.array([0.623, 0.0, -0.782])


@dataclass
class Box:
    pose: Pose2D            # centre of the footprint; yaw along its depth
    depth: float
    width: float
    height: float
    kind: int = OTHER


@dataclass
class Scene:
    station: Pose2D = None
    model: StationModel = StationModel()
    wall: tuple = None                  # (point (x, y), outward normal (x, y))
    boxes: list = field(default_factory=list)
    legs: list = field(default_factory=list)      # (x, y, radius, height)


def _hit_box(origins, dirs, box):
    """Distance along each ray to the box (inf on a miss), and the face hit."""
    c, s = math.cos(box.pose.yaw), math.sin(box.pose.yaw)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])   # fixed -> box frame
    o = (origins - np.array([box.pose.x, box.pose.y, 0.0])) @ rot.T
    d = dirs @ rot.T
    lo = np.array([-box.depth / 2, -box.width / 2, 0.0])
    hi = np.array([box.depth / 2, box.width / 2, box.height])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (lo - o) / d, (hi - o) / d
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    t_enter, t_exit = np.nanmax(tmin, axis=1), np.nanmin(tmax, axis=1)
    axis = np.nanargmax(tmin, axis=1)
    hit = (t_enter <= t_exit) & (t_enter > 1e-6)
    t = np.where(hit, t_enter, np.inf)
    face = np.full(len(t), OTHER)
    if box.kind == STATION_TOP:
        face = np.where(axis == 2, STATION_TOP, np.where(axis == 0, STATION_FRONT, STATION_SIDE))
    return t, face


def _hit_leg(origins, dirs, leg):
    x, y, r, h = leg
    ox, oy = origins[:, 0] - x, origins[:, 1] - y
    a = dirs[:, 0] ** 2 + dirs[:, 1] ** 2
    b = 2 * (ox * dirs[:, 0] + oy * dirs[:, 1])
    c = ox ** 2 + oy ** 2 - r ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid='ignore', divide='ignore'):
        t = (-b - np.sqrt(disc)) / (2 * a)
    z = origins[:, 2] + t * dirs[:, 2]
    ok = (disc >= 0) & (t > 1e-6) & (z >= 0) & (z <= h)
    return np.where(ok, t, np.inf)


def cast(scene, origins, dirs):
    """Nearest hit of each ray: (points (N, 3), kind (N,)); NaN rows for misses."""
    n = len(dirs)
    best = np.full(n, np.inf)
    kind = np.full(n, -1)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_floor = -origins[:, 2] / dirs[:, 2]
    t_floor = np.where(t_floor > 1e-6, t_floor, np.inf)
    best, kind = t_floor, np.where(np.isfinite(t_floor), FLOOR, -1)
    candidates = []
    if scene.station is not None:
        m = scene.model
        candidates.append(_hit_box(origins, dirs,
                                   Box(scene.station, m.depth, m.width, m.height, STATION_TOP)))
    for box in scene.boxes:
        candidates.append(_hit_box(origins, dirs, box))
    for leg in scene.legs:
        candidates.append((_hit_leg(origins, dirs, leg), np.full(n, OTHER)))
    if scene.wall is not None:
        p0, normal = np.array([*scene.wall[0], 0.0]), np.array([*scene.wall[1], 0.0])
        with np.errstate(divide='ignore', invalid='ignore'):
            t_wall = ((p0 - origins) @ normal) / (dirs @ normal)
        candidates.append((np.where(t_wall > 1e-6, t_wall, np.inf), np.full(n, WALL)))
    for t, face in candidates:
        closer = t < best
        best = np.where(closer, t, best)
        kind = np.where(closer, face, kind)
    points = origins + best[:, None] * dirs
    points[~np.isfinite(best)] = np.nan
    return points, kind


def robot_at(station, distance_to_front, model=StationModel(), yaw_error=0.0):
    """Base pose facing the station, its lidar `distance_to_front` from the front face."""
    back = model.depth / 2 + distance_to_front + LIDAR_IN_BASE[0]
    x, y = from_frame(-back, 0.0, station)
    return Pose2D(x, y, station.yaw + yaw_error)


def lidar_scan(scene, robot, n_rays=120_000, noise=0.01, seed=0):
    """Fixed-frame (N, 3) L2 points: front hemisphere, range noise along each ray."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_rays, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs[:, 0] = np.abs(dirs[:, 0])                         # dome axis points forward
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    dirs = dirs @ rot.T
    origin = np.array([robot.x, robot.y, 0.0]) + rot @ LIDAR_IN_BASE
    points, kind = cast(scene, np.tile(origin, (n_rays, 1)), dirs)
    ok = np.isfinite(points).all(axis=1) & (np.linalg.norm(points - origin, axis=1) < 6.0)
    points, dirs = points[ok], dirs[ok]
    return points + rng.normal(scale=noise, size=(len(points), 1)) * dirs


def bottom_camera(robot):
    """Camera (dock_perception.Camera) for the bottom camera of a robot at `robot`."""
    c, s = math.cos(robot.yaw), math.sin(robot.yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    z_axis = rot @ (BOTTOM_CAMERA_AXIS / np.linalg.norm(BOTTOM_CAMERA_AXIS))
    x_axis = rot @ np.array([0.0, -1.0, 0.0])
    y_axis = np.cross(z_axis, x_axis)
    position = np.array([robot.x, robot.y, 0.0]) + rot @ BOTTOM_CAMERA_IN_BASE
    return Camera(BOTTOM_CAMERA_K, BOTTOM_CAMERA_D,
                  np.stack([x_axis, y_axis, z_axis], axis=1), position)


# BGR shades: a single-colour (blue-grey) floor, the white station's faces.
COLOURS = {FLOOR: (110, 100, 90), STATION_TOP: (240, 240, 240),
           STATION_FRONT: (215, 215, 215), STATION_SIDE: (225, 225, 225),
           WALL: (235, 235, 235), OTHER: (60, 60, 60)}


def render(scene, camera, shape=(480, 640), brightness=1.0, noise=4.0, seed=0, colours=None):
    """A BGR image of the scene from `camera`, with sensor noise."""
    import cv2
    colours = {**COLOURS, **(colours or {})}
    h, w = shape
    v, u = np.mgrid[0:h, 0:w]
    pixels = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float64)
    norm = cv2.undistortPoints(pixels.reshape(-1, 1, 2), camera.k, camera.d).reshape(-1, 2)
    dirs = np.hstack([norm, np.ones((len(norm), 1))]) @ camera.rotation.T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    _, kind = cast(scene, np.tile(camera.position, (len(dirs), 1)), dirs)
    image = np.zeros((h * w, 3))
    for k, colour in colours.items():
        image[kind == k] = colour
    rng = np.random.default_rng(seed)
    image = image * brightness + rng.normal(scale=noise, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8).reshape(h, w, 3)
