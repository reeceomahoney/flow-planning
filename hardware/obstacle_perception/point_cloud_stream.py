from __future__ import annotations

import importlib
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from camera_web import LiveState

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SceneProjection:
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    scale: float
    offset: np.ndarray
    table_origin: np.ndarray
    table_x: np.ndarray
    table_y: np.ndarray
    table_normal: np.ndarray
    table_bounds: tuple[float, float, float, float]
    table_inliers: int
    table_residual_mm: float


@dataclass(frozen=True)
class ObstacleBox:
    center_u: float
    center_v: float
    side: float
    bottom: float
    top: float
    points_used: int


BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
POLICY_COLORS = (
    (255, 108, 62),
    (62, 190, 255),
    (226, 80, 242),
)
VISUAL_HULL_STEP = 0.005
VISUAL_HULL_HALF_WIDTH = 0.12
VISUAL_HULL_MAX_HEIGHT = 0.4


def load_calibration(
    path: Path,
    top_serial: str,
    left_serial: str,
) -> tuple[np.ndarray, str]:
    if not path.is_file():
        raise RuntimeError(f"Camera calibration does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("reference_frame") != "top_camera_optical_frame":
        raise RuntimeError("Calibration reference frame is not the top camera")
    if payload.get("source_frame") != "left_wrist_camera_optical_frame":
        raise RuntimeError("Calibration source frame is not the left camera")
    if payload.get("top_serial") != top_serial:
        raise RuntimeError("Calibration top-camera serial does not match")
    if payload.get("left_serial") != left_serial:
        raise RuntimeError("Calibration left-camera serial does not match")
    transform = np.asarray(payload.get("top_from_left"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("Calibration transform is invalid")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise RuntimeError("Calibration transform is not rigid")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-3):
        raise RuntimeError("Calibration rotation is invalid")
    residual = payload.get("translation_residual_mm", {})
    median = float(residual.get("median", float("nan")))
    label = (
        f"{int(payload.get('inliers', 0))}/{int(payload.get('samples', 0))} poses"
        f" | median {median:.1f} mm"
    )
    return transform.astype(np.float32), label


def load_robot_calibration(
    path: Path,
    top_serial: str,
) -> tuple[np.ndarray, str, str]:
    if not path.is_file():
        raise RuntimeError(f"Robot calibration does not exist: {path}")
    payload = json.loads(path.read_text())
    if payload.get("reference_frame") != "piper_right_base_frame":
        raise RuntimeError("Robot calibration reference is not the right Piper base")
    if payload.get("source_frame") != "top_camera_optical_frame":
        raise RuntimeError("Robot calibration source is not the top camera")
    if payload.get("top_serial") != top_serial:
        raise RuntimeError("Robot calibration top-camera serial does not match")
    transform = np.asarray(payload.get("base_from_top"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise RuntimeError("Robot calibration transform is invalid")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0)):
        raise RuntimeError("Robot calibration transform is not rigid")
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-3):
        raise RuntimeError("Robot calibration rotation is invalid")
    residual = payload.get("translation_residual_mm", {})
    median = float(residual.get("median", float("nan")))
    label = (
        f"robot {int(payload.get('inliers', 0))}/{int(payload.get('samples', 0))}"
        f" | {median:.1f} mm"
    )
    return transform.astype(np.float32), label, str(payload.get("can_interface"))


def transform_from_rpy(rpy: np.ndarray, translation: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    rotation_y = np.asarray(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rotation_z = np.asarray(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = translation
    return transform


def import_piper_sdk() -> Any:
    try:
        return importlib.import_module("piper_sdk")
    except ModuleNotFoundError:
        site_packages = sorted(
            (REPO_ROOT / ".pixi" / "envs" / "default" / "lib").glob(
                "python*/site-packages"
            )
        )
        if not site_packages:
            raise RuntimeError("Piper SDK is unavailable") from None
        sys.path.append(str(site_packages[-1]))
        return importlib.import_module("piper_sdk")


class PiperPoseReader:
    def __init__(self, can_interface: str):
        self.can_interface = can_interface
        self.interface: Any = None
        self.error: str | None = None

    def connect(self) -> None:
        try:
            if not (Path("/sys/class/net") / self.can_interface).exists():
                raise RuntimeError(f"{self.can_interface} is unavailable")
            module = import_piper_sdk()
            self.interface = module.C_PiperInterface_V2(self.can_interface)
            self.interface.ConnectPort(piper_init=False)
            self.error = None
        except Exception as error:
            self.interface = None
            self.error = str(error)

    def read(self) -> np.ndarray | None:
        if self.interface is None:
            return None
        message = self.interface.GetArmEndPoseMsgs()
        if float(message.time_stamp) <= 0.0 or float(message.Hz) <= 0.0:
            return None
        pose = message.end_pose
        translation = (
            np.asarray(
                [pose.X_axis, pose.Y_axis, pose.Z_axis],
                dtype=np.float64,
            )
            / 1_000_000.0
        )
        rpy = np.radians(
            np.asarray(
                [pose.RX_axis, pose.RY_axis, pose.RZ_axis],
                dtype=np.float64,
            )
            / 1000.0
        )
        return transform_from_rpy(rpy, translation).astype(np.float32)

    def read_state(self) -> tuple[float, ...] | None:
        if self.interface is None:
            return None
        joint_message = self.interface.GetArmJointMsgs()
        gripper_message = self.interface.GetArmGripperMsgs()
        if (
            float(joint_message.time_stamp) <= 0.0
            or float(gripper_message.time_stamp) <= 0.0
        ):
            return None
        joints = joint_message.joint_state
        state = [
            float(getattr(joints, f"joint_{index}")) / 1000.0 for index in range(1, 7)
        ]
        state.append(float(gripper_message.gripper_state.grippers_angle) / 10000.0)
        return tuple(state)

    def disconnect(self) -> None:
        if self.interface is not None:
            self.interface.DisconnectPort()
            self.interface = None


def camera_points(
    sample: dict[str, object], max_points: int, trim_percent: float
) -> tuple[np.ndarray, int, int]:
    depth_m = cast(np.ndarray, sample["depth_m"])
    mask = cast(np.ndarray, sample["mask"]).astype(bool)
    intrinsics = cast(dict[str, Any], sample["intrinsics"])
    if mask.shape != depth_m.shape:
        raise RuntimeError("Depth and mask dimensions differ")
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 3.0)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return np.empty((0, 3), np.float32), 0, 0
    z = depth_m[rows, columns]
    x = (
        (columns.astype(np.float32) - float(intrinsics["ppx"]))
        * z
        / float(intrinsics["fx"])
    )
    y = (
        (rows.astype(np.float32) - float(intrinsics["ppy"]))
        * z
        / float(intrinsics["fy"])
    )
    points = np.column_stack((x, y, z))
    total = len(points)
    points = trim_outliers(points, trim_percent)
    kept = len(points)
    if kept > max_points:
        indices = np.linspace(0, kept - 1, max_points, dtype=np.int64)
        points = points[indices]
    return points, total, kept


def trim_outliers(points: np.ndarray, percentage: float) -> np.ndarray:
    if percentage <= 0.0 or len(points) < 20:
        return points
    keep = max(4, round(len(points) * (1.0 - percentage / 100.0)))
    center = np.median(points, axis=0)
    deviations = np.abs(points - center)
    scale = 1.4826 * np.median(deviations, axis=0)
    scale = np.maximum(scale, 0.002)
    scores = np.linalg.norm((points - center) / scale, axis=1)
    indices = np.argpartition(scores, keep - 1)[:keep]
    return points[indices]


def context_points(sample: dict[str, object], max_points: int) -> np.ndarray:
    depth_m = cast(np.ndarray, sample["depth_m"])
    mask = cast(np.ndarray, sample["mask"]).astype(bool)
    intrinsics = cast(dict[str, Any], sample["intrinsics"])
    valid = (~mask) & np.isfinite(depth_m) & (depth_m > 0.12) & (depth_m < 1.5)
    rows, columns = np.nonzero(valid)
    if not len(rows):
        return np.empty((0, 3), np.float32)
    if len(rows) > max_points:
        indices = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows = rows[indices]
        columns = columns[indices]
    z = depth_m[rows, columns]
    x = (
        (columns.astype(np.float32) - float(intrinsics["ppx"]))
        * z
        / float(intrinsics["fx"])
    )
    y = (
        (rows.astype(np.float32) - float(intrinsics["ppy"]))
        * z
        / float(intrinsics["fy"])
    )
    return np.column_stack((x, y, z))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if not len(points):
        return points.copy()
    return points @ transform[:3, :3].T + transform[:3, 3]


def estimate_obstacle_box(
    top_points: np.ndarray,
    left_points: np.ndarray,
    top_from_left: np.ndarray,
    projection: SceneProjection,
    center_points: np.ndarray,
) -> ObstacleBox | None:
    clouds = []
    if len(top_points):
        clouds.append(top_points)
    if len(left_points):
        clouds.append(transform_points(left_points, top_from_left))
    if not clouds:
        return None
    points = np.vstack(clouds)
    relative = points - projection.table_origin
    table_u = relative @ projection.table_x
    table_v = relative @ projection.table_y
    heights = relative @ projection.table_normal
    valid = (
        np.isfinite(table_u)
        & np.isfinite(table_v)
        & np.isfinite(heights)
        & (heights > 0.005)
        & (heights < 0.5)
    )
    table_u = table_u[valid]
    table_v = table_v[valid]
    heights = heights[valid]
    if len(heights) < 100:
        return None
    low_u, high_u = np.percentile(table_u, (1.5, 98.5))
    low_v, high_v = np.percentile(table_v, (1.5, 98.5))
    center_u = float(low_u + high_u) * 0.5
    center_v = float(low_v + high_v) * 0.5
    center_relative = center_points - projection.table_origin
    center_u_values = center_relative @ projection.table_x
    center_v_values = center_relative @ projection.table_y
    center_heights = center_relative @ projection.table_normal
    center_valid = (
        np.isfinite(center_u_values)
        & np.isfinite(center_v_values)
        & np.isfinite(center_heights)
        & (center_heights > 0.005)
        & (center_heights < 0.5)
    )
    if np.count_nonzero(center_valid) >= 100:
        center_u = float(np.median(center_u_values[center_valid]))
        center_v = float(np.median(center_v_values[center_valid]))
    observed_side = max(float(high_u - low_u), float(high_v - low_v))
    side = observed_side + 0.008
    top = float(np.percentile(heights, 99.0)) + 0.006
    return ObstacleBox(center_u, center_v, side, 0.0, top, len(heights))


def clean_silhouette(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    rows, columns = np.nonzero(labels == largest)
    if len(rows) < 20:
        return np.zeros_like(mask, dtype=bool)
    hull = cv2.convexHull(np.column_stack((columns, rows)).astype(np.int32))
    filled = np.zeros_like(binary)
    cv2.fillConvexPoly(filled, hull, 1, cv2.LINE_AA)
    return cv2.dilate(filled, np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def points_inside_silhouette(
    points_base: np.ndarray,
    camera_from_base: np.ndarray,
    intrinsics: dict[str, Any],
    silhouette: np.ndarray,
) -> np.ndarray:
    points_camera = transform_points(points_base, camera_from_base)
    z = points_camera[:, 2]
    valid = np.isfinite(points_camera).all(axis=1) & (z > 0.05)
    columns = np.zeros(len(points_camera), dtype=np.int32)
    rows = np.zeros(len(points_camera), dtype=np.int32)
    columns[valid] = np.rint(
        float(intrinsics["fx"]) * points_camera[valid, 0] / points_camera[valid, 2]
        + float(intrinsics["ppx"])
    ).astype(np.int32)
    rows[valid] = np.rint(
        float(intrinsics["fy"]) * points_camera[valid, 1] / points_camera[valid, 2]
        + float(intrinsics["ppy"])
    ).astype(np.int32)
    valid &= (
        (columns >= 0)
        & (columns < silhouette.shape[1])
        & (rows >= 0)
        & (rows < silhouette.shape[0])
    )
    inside = np.zeros(len(points_base), dtype=bool)
    indices = np.flatnonzero(valid)
    inside[indices] = silhouette[rows[indices], columns[indices]]
    return inside


def estimate_visual_hull_box(
    top_sample: dict[str, object],
    left_sample: dict[str, object],
    projection: SceneProjection,
    top_from_base: np.ndarray,
    left_from_base: np.ndarray,
    seed_points_base: np.ndarray,
) -> ObstacleBox | None:
    if len(seed_points_base) < 100:
        return None
    relative = seed_points_base - projection.table_origin
    seed_u = relative @ projection.table_x
    seed_v = relative @ projection.table_y
    seed_height = relative @ projection.table_normal
    valid = (
        np.isfinite(seed_u)
        & np.isfinite(seed_v)
        & np.isfinite(seed_height)
        & (seed_height > 0.005)
        & (seed_height < VISUAL_HULL_MAX_HEIGHT)
    )
    if np.count_nonzero(valid) < 100:
        return None
    center_u = float(np.median(seed_u[valid]))
    center_v = float(np.median(seed_v[valid]))
    minimum_u, maximum_u, minimum_v, maximum_v = projection.table_bounds
    coordinates_u = np.arange(
        max(minimum_u, center_u - VISUAL_HULL_HALF_WIDTH),
        min(maximum_u, center_u + VISUAL_HULL_HALF_WIDTH) + VISUAL_HULL_STEP * 0.5,
        VISUAL_HULL_STEP,
        dtype=np.float32,
    )
    coordinates_v = np.arange(
        max(minimum_v, center_v - VISUAL_HULL_HALF_WIDTH),
        min(maximum_v, center_v + VISUAL_HULL_HALF_WIDTH) + VISUAL_HULL_STEP * 0.5,
        VISUAL_HULL_STEP,
        dtype=np.float32,
    )
    heights = np.arange(
        VISUAL_HULL_STEP * 0.5,
        VISUAL_HULL_MAX_HEIGHT,
        VISUAL_HULL_STEP,
        dtype=np.float32,
    )
    if not len(coordinates_u) or not len(coordinates_v):
        return None
    grid_u, grid_v, grid_height = np.meshgrid(
        coordinates_u,
        coordinates_v,
        heights,
        indexing="ij",
    )
    points_base = (
        projection.table_origin
        + grid_u.reshape(-1, 1) * projection.table_x
        + grid_v.reshape(-1, 1) * projection.table_y
        + grid_height.reshape(-1, 1) * projection.table_normal
    ).astype(np.float32)
    top_mask = clean_silhouette(cast(np.ndarray, top_sample["mask"]))
    left_mask = clean_silhouette(cast(np.ndarray, left_sample["mask"]))
    if not top_mask.any() or not left_mask.any():
        return None
    top_intrinsics = cast(dict[str, Any], top_sample["intrinsics"])
    left_intrinsics = cast(dict[str, Any], left_sample["intrinsics"])
    occupied = points_inside_silhouette(
        points_base,
        top_from_base,
        top_intrinsics,
        top_mask,
    ) & points_inside_silhouette(
        points_base,
        left_from_base,
        left_intrinsics,
        left_mask,
    )
    occupancy = occupied.reshape(
        len(coordinates_u),
        len(coordinates_v),
        len(heights),
    )
    cross_section_counts = np.count_nonzero(occupancy, axis=(0, 1))
    minimum_cross_section = max(16, round(float(cross_section_counts.max()) * 0.3))
    valid_height_indices = np.flatnonzero(cross_section_counts >= minimum_cross_section)
    if not len(valid_height_indices):
        return None
    top_index = int(valid_height_indices[-1])
    footprint = np.any(
        occupancy[:, :, max(0, top_index - 2) : top_index + 1],
        axis=2,
    ).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        footprint,
        connectivity=8,
    )
    if component_count <= 1:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    footprint = labels == largest
    footprint_rows, footprint_columns = np.nonzero(footprint)
    if len(footprint_rows) < 16:
        return None
    low_u = float(coordinates_u[footprint_rows].min() - VISUAL_HULL_STEP * 0.5)
    high_u = float(coordinates_u[footprint_rows].max() + VISUAL_HULL_STEP * 0.5)
    low_v = float(coordinates_v[footprint_columns].min() - VISUAL_HULL_STEP * 0.5)
    high_v = float(coordinates_v[footprint_columns].max() + VISUAL_HULL_STEP * 0.5)
    side = max(high_u - low_u, high_v - low_v)
    top = float(heights[top_index] + VISUAL_HULL_STEP * 0.5)
    if not 0.02 <= side <= 0.2 or not 0.04 <= top <= VISUAL_HULL_MAX_HEIGHT:
        return None
    return ObstacleBox(
        (low_u + high_u) * 0.5,
        (low_v + high_v) * 0.5,
        side,
        0.0,
        top,
        int(np.count_nonzero(occupancy[:, :, : top_index + 1])),
    )


def smooth_obstacle_box(
    previous: ObstacleBox | None,
    current: ObstacleBox | None,
) -> ObstacleBox | None:
    if current is None:
        return previous
    if previous is None:
        return current
    movement = np.hypot(
        current.center_u - previous.center_u,
        current.center_v - previous.center_v,
    )
    center_alpha = 0.65 if movement > 0.05 else 0.28
    size_alpha = 0.14
    height_alpha = 0.18
    return ObstacleBox(
        previous.center_u + center_alpha * (current.center_u - previous.center_u),
        previous.center_v + center_alpha * (current.center_v - previous.center_v),
        previous.side + size_alpha * (current.side - previous.side),
        0.0,
        previous.top + height_alpha * (current.top - previous.top),
        current.points_used,
    )


def obstacle_box_corners(
    box: ObstacleBox,
    projection: SceneProjection,
) -> np.ndarray:
    half = box.side * 0.5
    corners = []
    for height in (box.bottom, box.top):
        for table_u, table_v in (
            (-half, -half),
            (half, -half),
            (half, half),
            (-half, half),
        ):
            corners.append(
                projection.table_origin
                + (box.center_u + table_u) * projection.table_x
                + (box.center_v + table_v) * projection.table_y
                + height * projection.table_normal
            )
    return np.asarray(corners, np.float32)


def project_points(
    points: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.empty((0, 2), np.float32), points
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0.05)
    visible = points[valid]
    if not len(visible):
        return np.empty((0, 2), np.float32), visible
    columns = float(intrinsics["fx"]) * visible[:, 0] / visible[:, 2] + float(
        intrinsics["ppx"]
    )
    rows = float(intrinsics["fy"]) * visible[:, 1] / visible[:, 2] + float(
        intrinsics["ppy"]
    )
    return np.column_stack((columns, rows)), visible


def project_gripper_marker(
    base_from_gripper: np.ndarray | None,
    camera_from_base: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[float, float, float] | None:
    if base_from_gripper is None:
        return None
    point_camera = transform_points(
        base_from_gripper[:3, 3].reshape(1, 3),
        camera_from_base,
    )[0]
    if not np.isfinite(point_camera).all() or float(point_camera[2]) <= 0.05:
        return None
    column = float(intrinsics["fx"]) * float(point_camera[0]) / float(
        point_camera[2]
    ) + float(intrinsics["ppx"])
    row = float(intrinsics["fy"]) * float(point_camera[1]) / float(
        point_camera[2]
    ) + float(intrinsics["ppy"])
    return column, row, float(point_camera[2])


def draw_gripper_marker(
    canvas: np.ndarray,
    marker: tuple[float, float, float] | None,
    source_width: float,
    source_height: float,
) -> None:
    if marker is None:
        return
    scale = min(canvas.shape[1] / source_width, canvas.shape[0] / source_height)
    offset_x = (canvas.shape[1] - source_width * scale) * 0.5
    offset_y = (canvas.shape[0] - source_height * scale) * 0.5
    radius = 10
    projected_center = (
        round(marker[0] * scale + offset_x),
        round(marker[1] * scale + offset_y),
    )
    center = (
        int(np.clip(projected_center[0], radius + 7, canvas.shape[1] - radius - 8)),
        int(np.clip(projected_center[1], radius + 7, canvas.shape[0] - radius - 8)),
    )
    direction = ""
    if projected_center[0] < 0:
        direction += "<"
    elif projected_center[0] >= canvas.shape[1]:
        direction += ">"
    if projected_center[1] < 0:
        direction += "^"
    elif projected_center[1] >= canvas.shape[0]:
        direction += "v"
    for color, thickness in (((6, 9, 14), 5), ((45, 220, 255), 2)):
        cv2.circle(canvas, center, radius, color, thickness, cv2.LINE_AA)
        cv2.line(
            canvas,
            (center[0] - radius - 5, center[1]),
            (center[0] + radius + 5, center[1]),
            color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (center[0], center[1] - radius - 5),
            (center[0], center[1] + radius + 5),
            color,
            thickness,
            cv2.LINE_AA,
        )
    label = (
        f"GRIPPER {direction} {marker[2]:.2f} m"
        if direction
        else (f"GRIPPER {marker[2]:.2f} m")
    )
    label_x = min(max(center[0] + 15, 5), canvas.shape[1] - 125)
    label_y = min(max(center[1] - 12, 18), canvas.shape[0] - 5)
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (6, 9, 14),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (45, 220, 255),
        1,
        cv2.LINE_AA,
    )


def rasterize_points(
    labels: np.ndarray,
    projected: np.ndarray,
    intrinsics: dict[str, Any],
    code: int,
) -> None:
    if not len(projected):
        return
    scale = min(
        labels.shape[1] / float(intrinsics["width"]),
        labels.shape[0] / float(intrinsics["height"]),
    )
    offset_x = (labels.shape[1] - float(intrinsics["width"]) * scale) * 0.5
    offset_y = (labels.shape[0] - float(intrinsics["height"]) * scale) * 0.5
    pixels_x = np.rint(projected[:, 0] * scale + offset_x).astype(int)
    pixels_y = np.rint(projected[:, 1] * scale + offset_y).astype(int)
    for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        x = pixels_x + offset_x
        y = pixels_y + offset_y
        inside = (
            (x >= 1) & (x < labels.shape[1] - 1) & (y >= 1) & (y < labels.shape[0] - 1)
        )
        np.bitwise_or.at(labels, (y[inside], x[inside]), code)


def empty_render(name: str, message: str) -> np.ndarray:
    canvas = np.full((480, 640, 3), (8, 11, 16), dtype=np.uint8)
    camera = "TOP" if name == "top" else "LEFT WRIST"
    cv2.putText(
        canvas,
        f"{camera} CAMERA POV",
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (190, 201, 216),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        message,
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (104, 116, 134),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_grid(canvas: np.ndarray) -> None:
    for x in range(40, canvas.shape[1], 40):
        cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (15, 21, 29), 1)
    for y in range(40, canvas.shape[0], 40):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (15, 21, 29), 1)


def draw_legend(canvas: np.ndarray) -> None:
    entries = (
        ("TOP", (255, 82, 38)),
        ("LEFT", (38, 67, 255)),
        ("OVERLAP", (236, 82, 236)),
    )
    x = 400
    for label, color in entries:
        cv2.circle(canvas, (x, 45), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (x + 8, 49),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (145, 157, 174),
            1,
            cv2.LINE_AA,
        )
        x += 75


def camera_background(rgb: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if image.shape[:2] != (480, 640):
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    return cv2.convertScaleAbs(image, alpha=0.38)


def blend_point_labels(canvas: np.ndarray, labels: np.ndarray) -> None:
    colors = (
        (1, np.asarray((255, 82, 38), dtype=np.float32)),
        (2, np.asarray((38, 67, 255), dtype=np.float32)),
        (3, np.asarray((236, 82, 236), dtype=np.float32)),
    )
    alpha = 0.82
    for code, color in colors:
        selected = labels == code
        if not np.any(selected):
            continue
        blended = canvas[selected].astype(np.float32) * (1.0 - alpha) + color * alpha
        canvas[selected] = np.rint(blended).astype(np.uint8)


def draw_camera_box(
    canvas: np.ndarray,
    corners: np.ndarray,
    intrinsics: dict[str, Any],
) -> None:
    projected = project_camera_box(corners, intrinsics)
    draw_projected_camera_box(
        canvas,
        projected,
        float(intrinsics["width"]),
        float(intrinsics["height"]),
    )


def project_camera_box(
    corners: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[tuple[float, float], ...] | None:
    projected, visible = project_points(corners, intrinsics)
    if len(visible) != 8:
        return None
    return tuple((float(point[0]), float(point[1])) for point in projected)


def draw_projected_camera_box(
    canvas: np.ndarray,
    projected: tuple[tuple[float, float], ...] | None,
    source_width: float,
    source_height: float,
) -> None:
    if projected is None or len(projected) != 8:
        return
    scale = min(
        canvas.shape[1] / source_width,
        canvas.shape[0] / source_height,
    )
    offset = np.asarray(
        (
            (canvas.shape[1] - source_width * scale) * 0.5,
            (canvas.shape[0] - source_height * scale) * 0.5,
        ),
        np.float32,
    )
    pixels = np.rint(np.asarray(projected) * scale + offset).astype(int)
    for start, end in BOX_EDGES:
        cv2.line(
            canvas,
            tuple(pixels[start]),
            tuple(pixels[end]),
            (5, 8, 12),
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            tuple(pixels[start]),
            tuple(pixels[end]),
            (255, 205, 65),
            2,
            cv2.LINE_AA,
        )


def project_policy_trajectories(
    trajectories: tuple[tuple[tuple[float, float, float], ...], ...],
    camera_from_base: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    projected_trajectories: list[tuple[tuple[float, float], ...]] = []
    for trajectory in trajectories:
        points = np.asarray(trajectory, dtype=np.float32)
        camera_points = transform_points(points, camera_from_base)
        projected, _ = project_points(camera_points, intrinsics)
        if len(projected) < 2:
            continue
        projected_trajectories.append(
            tuple((float(point[0]), float(point[1])) for point in projected)
        )
    return tuple(projected_trajectories)


def draw_projected_policy_trajectories(
    canvas: np.ndarray,
    trajectories: tuple[tuple[tuple[float, float], ...], ...],
    source_width: float,
    source_height: float,
    pickup_steps: tuple[int, ...] = (),
) -> None:
    scale = min(
        canvas.shape[1] / source_width,
        canvas.shape[0] / source_height,
    )
    offset = np.asarray(
        (
            (canvas.shape[1] - source_width * scale) * 0.5,
            (canvas.shape[0] - source_height * scale) * 0.5,
        ),
        dtype=np.float32,
    )
    pickup_pixels: list[tuple[int, int, tuple[int, int, int]]] = []
    for index, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
        color = POLICY_COLORS[index % len(POLICY_COLORS)]
        pixels = np.rint(np.asarray(trajectory) * scale + offset).astype(np.int32)
        shadow = pixels + (3, 4)
        dark = tuple(round(channel * 0.22) for channel in color)
        highlight = tuple(min(255, channel + 105) for channel in color)
        cv2.polylines(canvas, [shadow], False, (2, 4, 7), 11, cv2.LINE_AA)
        cv2.polylines(canvas, [pixels], False, dark, 10, cv2.LINE_AA)
        cv2.polylines(canvas, [pixels], False, color, 7, cv2.LINE_AA)
        cv2.polylines(
            canvas,
            [pixels + (-1, -1)],
            False,
            highlight,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, tuple(pixels[0]), 6, (235, 241, 250), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[-1]), 7, dark, 3, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[-1]), 5, color, -1, cv2.LINE_AA)
        if index < len(pickup_steps) and pickup_steps[index] < len(pixels):
            pickup = tuple(pixels[pickup_steps[index]])
            pickup_pixels.append((pickup[0], pickup[1], color))
    if not pickup_pixels:
        return
    for column, row, color in pickup_pixels:
        cv2.circle(canvas, (column, row), 10, (2, 4, 7), -1, cv2.LINE_AA)
        cv2.circle(canvas, (column, row), 7, (245, 249, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (column, row), 4, color, -1, cv2.LINE_AA)
    pickup_center = np.rint(
        np.asarray([(point[0], point[1]) for point in pickup_pixels]).mean(axis=0)
    ).astype(int)
    label_position = (int(pickup_center[0]) + 12, int(pickup_center[1]) - 10)
    cv2.putText(
        canvas,
        "PICKUP",
        (label_position[0] + 1, label_position[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (2, 4, 7),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "PICKUP",
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (30, 220, 255),
        1,
        cv2.LINE_AA,
    )


def render_shared_cloud(
    viewpoint: str,
    background_rgb: np.ndarray,
    top_points: np.ndarray,
    top_total: int,
    top_kept: int,
    top_intrinsics: dict[str, Any],
    left_points: np.ndarray,
    left_total: int,
    left_kept: int,
    left_intrinsics: dict[str, Any],
    top_from_left: np.ndarray,
    calibration_label: str,
    trim_percent: float,
    box_corners_top: np.ndarray | None,
    gripper_marker: tuple[float, float, float] | None,
    policy_trajectories: tuple[tuple[tuple[float, float], ...], ...],
    pickup_steps: tuple[int, ...],
) -> np.ndarray:
    if not len(top_points) and not len(left_points):
        return empty_render(viewpoint, "No valid tracked depth points")
    if viewpoint == "top":
        blue_points = top_points
        red_points = transform_points(left_points, top_from_left)
        intrinsics = top_intrinsics
        box_corners = box_corners_top
    elif viewpoint == "left":
        left_from_top = np.linalg.inv(top_from_left)
        blue_points = transform_points(top_points, left_from_top)
        red_points = left_points
        intrinsics = left_intrinsics
        box_corners = (
            None
            if box_corners_top is None
            else transform_points(box_corners_top, left_from_top)
        )
    else:
        raise ValueError(f"Unknown viewpoint: {viewpoint}")
    viewpoint_label = "TOP" if viewpoint == "top" else "LEFT WRIST"
    blue_projected, blue_visible = project_points(blue_points, intrinsics)
    red_projected, red_visible = project_points(red_points, intrinsics)
    projections = [item for item in (blue_projected, red_projected) if len(item)]
    if not projections:
        return empty_render(viewpoint, "Tracked points are behind this camera")
    canvas = camera_background(background_rgb)
    labels = np.zeros(canvas.shape[:2], dtype=np.uint8)
    rasterize_points(labels, blue_projected, intrinsics, 1)
    rasterize_points(labels, red_projected, intrinsics, 2)
    blend_point_labels(canvas, labels)
    draw_projected_policy_trajectories(
        canvas,
        policy_trajectories,
        float(intrinsics["width"]),
        float(intrinsics["height"]),
        pickup_steps,
    )
    if box_corners is not None:
        draw_camera_box(canvas, box_corners, intrinsics)
    draw_gripper_marker(
        canvas,
        gripper_marker,
        float(intrinsics["width"]),
        float(intrinsics["height"]),
    )
    overlap = int(np.count_nonzero(labels == 3))
    blue_pixels = int(np.count_nonzero(labels & 1))
    red_pixels = int(np.count_nonzero(labels & 2))
    overlap_percent = 100.0 * overlap / max(min(blue_pixels, red_pixels), 1)
    visible_sets = [item for item in (blue_visible, red_visible) if len(item)]
    depths = np.concatenate([item[:, 2] for item in visible_sets])
    shade = canvas.copy()
    cv2.rectangle(shade, (0, 0), (canvas.shape[1], 58), (0, 0, 0), -1)
    cv2.rectangle(
        shade,
        (0, canvas.shape[0] - 28),
        (canvas.shape[1], canvas.shape[0]),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(shade, 0.62, canvas, 0.38, 0.0, canvas)
    cv2.putText(
        canvas,
        f"MERGED CLOUD | {viewpoint_label} CAMERA POV",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (202, 211, 224),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"blue top {top_kept:,}/{top_total:,} | red left "
            f"{left_kept:,}/{left_total:,} | trim {trim_percent:.0f}%"
            f" | overlap {overlap_percent:.0f}%"
        ),
        (18, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (113, 128, 148),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"depth {np.percentile(depths, 1):.3f}-{np.percentile(depths, 99):.3f} m",
        (445, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (113, 128, 148),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        calibration_label,
        (18, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (88, 107, 132),
        1,
        cv2.LINE_AA,
    )
    draw_legend(canvas)
    return canvas


def create_scene_projection(
    points: np.ndarray,
    base_from_top: np.ndarray,
    base_from_left: np.ndarray,
    canvas_shape: tuple[int, int],
    top_intrinsics: dict[str, Any],
    left_intrinsics: dict[str, Any],
) -> SceneProjection:
    if len(points) < 100:
        raise RuntimeError("Not enough scene depth points")
    sample_count = min(len(points), 8000)
    sample_indices = np.linspace(0, len(points) - 1, sample_count, dtype=np.int64)
    sample = points[sample_indices]
    random = np.random.default_rng(7)
    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = 0
    for _ in range(120):
        selected = sample[random.choice(sample_count, 3, replace=False)]
        normal = np.cross(selected[1] - selected[0], selected[2] - selected[0])
        length = float(np.linalg.norm(normal))
        if length < 1e-6:
            continue
        normal /= length
        if abs(float(normal[2])) < 0.7:
            continue
        offset = float(normal @ selected[0])
        count = int(np.count_nonzero(np.abs(sample @ normal - offset) < 0.008))
        if count > best_count:
            best_normal = normal
            best_offset = offset
            best_count = count
    if best_normal is None or best_count < 100:
        raise RuntimeError("Could not find the table plane")
    inliers = points[np.abs(points @ best_normal - best_offset) < 0.01]
    origin = np.median(inliers, axis=0)
    _, _, vectors = np.linalg.svd(inliers - origin, full_matrices=False)
    normal = vectors[-1].astype(np.float32)
    if float(normal[2]) < 0.0:
        normal *= -1.0
    distances = np.abs((points - origin) @ normal)
    inliers = points[distances < 0.01]
    if len(inliers) < 100:
        raise RuntimeError("Table plane has too few inliers")
    origin = np.median(inliers, axis=0).astype(np.float32)
    residual_mm = float(np.median(np.abs((inliers - origin) @ normal)) * 1000.0)
    table_x = np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
    table_x -= normal * float(table_x @ normal)
    if float(np.linalg.norm(table_x)) < 0.2:
        table_x = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        table_x -= normal * float(table_x @ normal)
    table_x /= np.linalg.norm(table_x)
    table_y = np.cross(normal, table_x)
    table_y /= np.linalg.norm(table_y)
    relative = inliers - origin
    table_u = relative @ table_x
    table_v = relative @ table_y
    low_u, high_u = np.percentile(table_u, (1.0, 99.0))
    low_v, high_v = np.percentile(table_v, (1.0, 99.0))
    view_from_scene = np.asarray((0.95, -1.15, 0.8), dtype=np.float32)
    view_from_scene /= np.linalg.norm(view_from_scene)
    forward = -view_from_scene
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    table_corners = np.asarray(
        [
            origin + low_u * table_x + low_v * table_y,
            origin + low_u * table_x + high_v * table_y,
            origin + high_u * table_x + low_v * table_y,
            origin + high_u * table_x + high_v * table_y,
        ],
        dtype=np.float32,
    )
    reference = np.vstack(
        (
            table_corners,
            frustum_points(base_from_top, top_intrinsics, 0.13),
            frustum_points(base_from_left, left_intrinsics, 0.13),
            np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (0.18, 0.0, 0.0),
                    (0.0, 0.18, 0.0),
                    (0.0, 0.0, 0.18),
                ],
                dtype=np.float32,
            ),
        )
    )
    horizontal = reference @ right
    vertical = -(reference @ up)
    low_horizontal, high_horizontal = horizontal.min(), horizontal.max()
    low_vertical, high_vertical = vertical.min(), vertical.max()
    range_horizontal = max(float(high_horizontal - low_horizontal), 0.2)
    range_vertical = max(float(high_vertical - low_vertical), 0.2)
    height, width = canvas_shape
    scale = min((width - 50) / range_horizontal, (height - 90) / range_vertical)
    middle_horizontal = float(low_horizontal + high_horizontal) * 0.5
    middle_vertical = float(low_vertical + high_vertical) * 0.5
    offset = np.asarray(
        (
            width * 0.5 - middle_horizontal * scale,
            (height + 48) * 0.5 - middle_vertical * scale,
        ),
        dtype=np.float32,
    )
    return SceneProjection(
        right,
        up,
        forward,
        scale,
        offset,
        origin,
        table_x,
        table_y,
        normal,
        (float(low_u), float(high_u), float(low_v), float(high_v)),
        len(inliers),
        residual_mm,
    )


def project_scene(
    points: np.ndarray,
    projection: SceneProjection,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = points @ projection.right
    vertical = -(points @ projection.up)
    pixels = (
        np.column_stack((horizontal, vertical)) * projection.scale + projection.offset
    )
    return pixels, points @ projection.forward


def scene_pixel(point: np.ndarray, projection: SceneProjection) -> tuple[int, int]:
    pixels, _ = project_scene(point.reshape(1, 3), projection)
    return tuple(np.rint(pixels[0]).astype(int))


def draw_scene_line(
    canvas: np.ndarray,
    projection: SceneProjection,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    cv2.line(
        canvas,
        scene_pixel(start, projection),
        scene_pixel(end, projection),
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_table_grid(canvas: np.ndarray, projection: SceneProjection) -> None:
    low_u, high_u, low_v, high_v = projection.table_bounds
    for table_u in np.linspace(low_u, high_u, 9):
        draw_scene_line(
            canvas,
            projection,
            projection.table_origin
            + table_u * projection.table_x
            + low_v * projection.table_y,
            projection.table_origin
            + table_u * projection.table_x
            + high_v * projection.table_y,
            (38, 51, 67),
        )
    for table_v in np.linspace(low_v, high_v, 9):
        draw_scene_line(
            canvas,
            projection,
            projection.table_origin
            + low_u * projection.table_x
            + table_v * projection.table_y,
            projection.table_origin
            + high_u * projection.table_x
            + table_v * projection.table_y,
            (38, 51, 67),
        )


def frustum_points(
    camera_to_top: np.ndarray,
    intrinsics: dict[str, Any],
    length: float,
) -> np.ndarray:
    width = float(intrinsics["width"])
    height = float(intrinsics["height"])
    corners = []
    for column, row in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)):
        corners.append(
            (
                (column - float(intrinsics["ppx"])) * length / float(intrinsics["fx"]),
                (row - float(intrinsics["ppy"])) * length / float(intrinsics["fy"]),
                length,
            )
        )
    local = np.vstack((np.zeros((1, 3), np.float32), np.asarray(corners, np.float32)))
    return transform_points(local, camera_to_top)


def draw_camera(
    canvas: np.ndarray,
    projection: SceneProjection,
    camera_to_top: np.ndarray,
    intrinsics: dict[str, Any],
    label: str,
    color: tuple[int, int, int],
) -> None:
    points = frustum_points(camera_to_top, intrinsics, 0.13)
    origin = points[0]
    for corner in points[1:]:
        draw_scene_line(canvas, projection, origin, corner, color)
    for index in range(4):
        draw_scene_line(
            canvas,
            projection,
            points[index + 1],
            points[(index + 1) % 4 + 1],
            color,
        )
    pixel = scene_pixel(origin, projection)
    cv2.circle(canvas, pixel, 4, color, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (pixel[0] + 7, pixel[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_transform_axes(
    canvas: np.ndarray,
    projection: SceneProjection,
    transform: np.ndarray,
    label: str,
    length: float,
    thickness: int,
) -> None:
    origin = transform[:3, 3]
    axes = (
        ("X", length * transform[:3, 0], (60, 70, 245)),
        ("Y", length * transform[:3, 1], (75, 220, 105)),
        ("Z", length * transform[:3, 2], (245, 145, 70)),
    )
    for axis_label, direction, color in axes:
        endpoint = origin + direction
        start_pixel = scene_pixel(origin, projection)
        end_pixel = scene_pixel(endpoint, projection)
        cv2.arrowedLine(
            canvas,
            start_pixel,
            end_pixel,
            color,
            thickness,
            cv2.LINE_AA,
            tipLength=0.2,
        )
        cv2.putText(
            canvas,
            axis_label,
            (end_pixel[0] + 3, end_pixel[1] - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )
    pixel = scene_pixel(origin, projection)
    cv2.circle(canvas, pixel, 5, (7, 10, 15), -1, cv2.LINE_AA)
    cv2.circle(canvas, pixel, 3, (225, 230, 238), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        label,
        (pixel[0] + 7, pixel[1] + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (210, 218, 229),
        1,
        cv2.LINE_AA,
    )


def draw_gripper(
    canvas: np.ndarray,
    projection: SceneProjection,
    base_from_gripper: np.ndarray,
) -> None:
    draw_transform_axes(
        canvas,
        projection,
        base_from_gripper,
        "RIGHT GRIPPER",
        0.07,
        2,
    )
    origin = base_from_gripper[:3, 3]
    pixel = scene_pixel(origin, projection)
    cv2.circle(canvas, pixel, 9, (8, 11, 16), 3, cv2.LINE_AA)
    cv2.circle(canvas, pixel, 6, (45, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"({origin[0]:+.3f}, {origin[1]:+.3f}, {origin[2]:+.3f}) m",
        (pixel[0] + 8, pixel[1] + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (45, 220, 255),
        1,
        cv2.LINE_AA,
    )


def rasterize_scene_points(
    labels: np.ndarray,
    points: np.ndarray,
    projection: SceneProjection,
    code: int,
) -> None:
    if not len(points):
        return
    pixels, _ = project_scene(points, projection)
    pixels = np.rint(pixels).astype(int)
    for offset_x, offset_y in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        x = pixels[:, 0] + offset_x
        y = pixels[:, 1] + offset_y
        inside = (
            (x >= 1) & (x < labels.shape[1] - 1) & (y >= 1) & (y < labels.shape[0] - 1)
        )
        np.bitwise_or.at(labels, (y[inside], x[inside]), code)


def draw_scene_box(
    canvas: np.ndarray,
    projection: SceneProjection,
    box: ObstacleBox,
) -> None:
    corners = obstacle_box_corners(box, projection)
    for start, end in BOX_EDGES:
        draw_scene_line(
            canvas,
            projection,
            corners[start],
            corners[end],
            (5, 8, 12),
            4,
        )
        draw_scene_line(
            canvas,
            projection,
            corners[start],
            corners[end],
            (45, 220, 255),
            2,
        )
    top_center = (
        projection.table_origin
        + box.center_u * projection.table_x
        + box.center_v * projection.table_y
        + box.top * projection.table_normal
    )
    pixel = scene_pixel(top_center, projection)
    cv2.putText(
        canvas,
        f"{box.side * 100:.1f} x {box.side * 100:.1f} x {box.top * 100:.1f} cm",
        (pixel[0] + 7, pixel[1] - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (45, 220, 255),
        1,
        cv2.LINE_AA,
    )


def draw_scene_policy_trajectories(
    canvas: np.ndarray,
    projection: SceneProjection,
    trajectories: tuple[tuple[tuple[float, float, float], ...], ...],
    pickup_steps: tuple[int, ...],
) -> None:
    projected_trajectories: list[tuple[tuple[float, float], ...]] = []
    for trajectory in trajectories:
        pixels, _ = project_scene(np.asarray(trajectory, dtype=np.float32), projection)
        projected_trajectories.append(
            tuple((float(point[0]), float(point[1])) for point in pixels)
        )
    draw_projected_policy_trajectories(
        canvas,
        tuple(projected_trajectories),
        float(canvas.shape[1]),
        float(canvas.shape[0]),
        pickup_steps,
    )


def render_oblique_scene(
    top_points_base: np.ndarray,
    top_total: int,
    top_kept: int,
    top_intrinsics: dict[str, Any],
    left_points_base: np.ndarray,
    left_total: int,
    left_kept: int,
    left_intrinsics: dict[str, Any],
    context_base: np.ndarray,
    base_from_top: np.ndarray,
    base_from_left: np.ndarray,
    base_from_gripper: np.ndarray | None,
    projection: SceneProjection,
    camera_calibration_label: str,
    robot_calibration_label: str,
    trim_percent: float,
    obstacle_box: ObstacleBox | None,
    policy_trajectories: tuple[tuple[tuple[float, float, float], ...], ...],
    pickup_steps: tuple[int, ...],
) -> np.ndarray:
    canvas = np.full((480, 640, 3), (8, 11, 16), dtype=np.uint8)
    draw_table_grid(canvas, projection)
    heights = (context_base - projection.table_origin) @ projection.table_normal
    raised = (heights > 0.012) & (heights < 0.6)
    scene_context = context_base[raised]
    scene_heights = heights[raised]
    context_pixels, context_depth = project_scene(scene_context, projection)
    context_pixels = np.rint(context_pixels).astype(int)
    order = np.argsort(context_depth)[::-1]
    context_pixels = context_pixels[order]
    context_height = np.clip(scene_heights[order] / 0.35, 0.0, 1.0)
    shades = np.rint(55.0 + 75.0 * context_height).astype(np.uint8)
    for offset_x, offset_y in ((0, 0), (1, 0), (0, 1)):
        pixels = context_pixels + (offset_x, offset_y)
        inside = (
            (pixels[:, 0] >= 1)
            & (pixels[:, 0] < canvas.shape[1] - 1)
            & (pixels[:, 1] >= 1)
            & (pixels[:, 1] < canvas.shape[0] - 1)
        )
        visible = pixels[inside]
        visible_shades = shades[inside]
        canvas[visible[:, 1], visible[:, 0]] = np.column_stack(
            (
                visible_shades,
                visible_shades,
                np.minimum(visible_shades + 8, 255),
            )
        )
    draw_camera(
        canvas,
        projection,
        base_from_top,
        top_intrinsics,
        "TOP CAM",
        (255, 82, 38),
    )
    draw_camera(
        canvas,
        projection,
        base_from_left,
        left_intrinsics,
        "LEFT CAM",
        (38, 67, 255),
    )
    labels = np.zeros(canvas.shape[:2], dtype=np.uint8)
    rasterize_scene_points(labels, top_points_base, projection, 1)
    rasterize_scene_points(labels, left_points_base, projection, 2)
    canvas[labels == 1] = (255, 82, 38)
    canvas[labels == 2] = (38, 67, 255)
    canvas[labels == 3] = (236, 82, 236)
    draw_scene_policy_trajectories(
        canvas,
        projection,
        policy_trajectories,
        pickup_steps,
    )
    if obstacle_box is not None:
        draw_scene_box(canvas, projection, obstacle_box)
    draw_transform_axes(
        canvas,
        projection,
        np.eye(4, dtype=np.float32),
        "RIGHT ROBOT BASE",
        0.16,
        2,
    )
    if base_from_gripper is not None:
        draw_gripper(canvas, projection, base_from_gripper)
    cv2.putText(
        canvas,
        "SHARED 3D SCENE | RIGHT ROBOT BASE FRAME",
        (18, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (202, 211, 224),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"blue top {top_kept:,}/{top_total:,} | red left "
            f"{left_kept:,}/{left_total:,} | trim {trim_percent:.0f}%"
        ),
        (18, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (113, 128, 148),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"table {projection.table_inliers:,} points | residual "
            f"{projection.table_residual_mm:.1f} mm | calibrated cameras"
        ),
        (18, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (88, 107, 132),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{camera_calibration_label} | {robot_calibration_label}",
        (356, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (88, 107, 132),
        1,
        cv2.LINE_AA,
    )
    draw_legend(canvas)
    return canvas


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("Could not encode point-cloud render")
    return encoded.tobytes()


class PointCloudStream:
    def __init__(
        self,
        live_state: LiveState,
        jpeg_quality: int,
        calibration_path: Path,
        robot_calibration_path: Path,
        top_serial: str,
        left_serial: str,
        can_interface: str,
    ):
        self.live_state = live_state
        self.jpeg_quality = jpeg_quality
        self.top_from_left, self.calibration_label = load_calibration(
            calibration_path,
            top_serial,
            left_serial,
        )
        (
            self.base_from_top,
            self.robot_calibration_label,
            calibrated_can_interface,
        ) = load_robot_calibration(robot_calibration_path, top_serial)
        self.base_from_left = self.base_from_top @ self.top_from_left
        self.top_from_base = np.linalg.inv(self.base_from_top)
        self.left_from_base = np.linalg.inv(self.base_from_left)
        self.pose_reader = PiperPoseReader(can_interface or calibrated_can_interface)
        self.scene_projection: SceneProjection | None = None
        self.obstacle_box: ObstacleBox | None = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.pose_reader.connect()
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.pose_reader.disconnect()

    def run(self) -> None:
        versions = {"top": -1, "left": -1}
        while not self.stop_event.is_set():
            top_current = self.live_state.sample("top")
            left_current = self.live_state.sample("left")
            if top_current is None or left_current is None:
                self.stop_event.wait(0.02)
                continue
            if (
                top_current[0] == versions["top"]
                and left_current[0] == versions["left"]
            ):
                self.stop_event.wait(0.02)
                continue
            versions = {"top": top_current[0], "left": left_current[0]}
            top_sample = top_current[1]
            left_sample = left_current[1]
            try:
                trim_percent = self.live_state.outlier_trim_percent()
                top_points, top_total, top_kept = camera_points(
                    top_sample,
                    10000,
                    trim_percent,
                )
                top_center_points, _, _ = camera_points(
                    top_sample,
                    10000,
                    0.0,
                )
                left_points, left_total, left_kept = camera_points(
                    left_sample,
                    10000,
                    trim_percent,
                )
                left_center_points, _, _ = camera_points(
                    left_sample,
                    10000,
                    0.0,
                )
                context = context_points(top_sample, 24000)
                top_points_base = transform_points(top_points, self.base_from_top)
                top_center_points_base = transform_points(
                    top_center_points,
                    self.base_from_top,
                )
                left_points_base = transform_points(left_points, self.base_from_left)
                left_center_points_base = transform_points(
                    left_center_points,
                    self.base_from_left,
                )
                context_base = transform_points(context, self.base_from_top)
                base_from_gripper = self.pose_reader.read()
                robot_state = self.pose_reader.read_state()
                if robot_state is not None:
                    self.live_state.publish_robot_state(robot_state)
                top_rgb = cast(np.ndarray, top_sample["rgb"])
                left_rgb = cast(np.ndarray, left_sample["rgb"])
                top_intrinsics = cast(dict[str, Any], top_sample["intrinsics"])
                left_intrinsics = cast(dict[str, Any], left_sample["intrinsics"])
                policy_trajectories = self.live_state.policy_trajectories()
                pickup_steps = self.live_state.policy_pickup_steps()
                top_policy_trajectories = project_policy_trajectories(
                    policy_trajectories,
                    self.top_from_base,
                    top_intrinsics,
                )
                left_policy_trajectories = project_policy_trajectories(
                    policy_trajectories,
                    self.left_from_base,
                    left_intrinsics,
                )
                self.live_state.publish_camera_trajectories(
                    "top",
                    top_policy_trajectories,
                )
                self.live_state.publish_camera_trajectories(
                    "left",
                    left_policy_trajectories,
                )
                top_gripper_marker = project_gripper_marker(
                    base_from_gripper,
                    self.top_from_base,
                    top_intrinsics,
                )
                left_gripper_marker = project_gripper_marker(
                    base_from_gripper,
                    self.left_from_base,
                    left_intrinsics,
                )
                self.live_state.publish_gripper_marker(
                    "top",
                    top_gripper_marker,
                )
                self.live_state.publish_gripper_marker(
                    "left",
                    left_gripper_marker,
                )
                if self.scene_projection is None:
                    self.scene_projection = create_scene_projection(
                        context_base,
                        self.base_from_top,
                        self.base_from_left,
                        (480, 640),
                        top_intrinsics,
                        left_intrinsics,
                    )
                point_estimate = estimate_obstacle_box(
                    top_points_base,
                    left_points_base,
                    np.eye(4, dtype=np.float32),
                    self.scene_projection,
                    top_center_points_base,
                )
                seed_points_base = np.vstack(
                    (top_center_points_base, left_center_points_base)
                )
                estimate = estimate_visual_hull_box(
                    top_sample,
                    left_sample,
                    self.scene_projection,
                    self.top_from_base,
                    self.left_from_base,
                    seed_points_base,
                )
                if estimate is None:
                    estimate = point_estimate
                self.obstacle_box = smooth_obstacle_box(self.obstacle_box, estimate)
                box_corners_base = (
                    None
                    if self.obstacle_box is None
                    else obstacle_box_corners(
                        self.obstacle_box,
                        self.scene_projection,
                    )
                )
                box_corners_top = (
                    None
                    if box_corners_base is None
                    else transform_points(box_corners_base, self.top_from_base)
                )
                box_corners_left = (
                    None
                    if box_corners_top is None
                    else transform_points(
                        box_corners_top,
                        np.linalg.inv(self.top_from_left),
                    )
                )
                self.live_state.publish_camera_box(
                    "top",
                    None
                    if box_corners_top is None
                    else project_camera_box(box_corners_top, top_intrinsics),
                )
                self.live_state.publish_camera_box(
                    "left",
                    None
                    if box_corners_left is None
                    else project_camera_box(box_corners_left, left_intrinsics),
                )
                top_render = render_shared_cloud(
                    "top",
                    top_rgb,
                    top_points,
                    top_total,
                    top_kept,
                    top_intrinsics,
                    left_points,
                    left_total,
                    left_kept,
                    left_intrinsics,
                    self.top_from_left,
                    self.calibration_label,
                    trim_percent,
                    box_corners_top,
                    top_gripper_marker,
                    top_policy_trajectories,
                    pickup_steps,
                )
                left_render = render_shared_cloud(
                    "left",
                    left_rgb,
                    top_points,
                    top_total,
                    top_kept,
                    top_intrinsics,
                    left_points,
                    left_total,
                    left_kept,
                    left_intrinsics,
                    self.top_from_left,
                    self.calibration_label,
                    trim_percent,
                    box_corners_top,
                    left_gripper_marker,
                    left_policy_trajectories,
                    pickup_steps,
                )
                scene_render = render_oblique_scene(
                    top_points_base,
                    top_total,
                    top_kept,
                    top_intrinsics,
                    left_points_base,
                    left_total,
                    left_kept,
                    left_intrinsics,
                    context_base,
                    self.base_from_top,
                    self.base_from_left,
                    base_from_gripper,
                    self.scene_projection,
                    self.calibration_label,
                    self.robot_calibration_label,
                    trim_percent,
                    self.obstacle_box,
                    policy_trajectories,
                    pickup_steps,
                )
            except Exception as error:
                top_render = empty_render("top", f"Point-cloud error: {error}")
                left_render = empty_render("left", f"Point-cloud error: {error}")
                scene_render = empty_render("left", f"Scene error: {error}")
            self.live_state.publish_image(
                "top_cloud.jpg",
                encode_jpeg(top_render, self.jpeg_quality),
            )
            self.live_state.publish_image(
                "left_cloud.jpg",
                encode_jpeg(left_render, self.jpeg_quality),
            )
            self.live_state.publish_image(
                "scene_cloud.jpg",
                encode_jpeg(scene_render, self.jpeg_quality),
            )
