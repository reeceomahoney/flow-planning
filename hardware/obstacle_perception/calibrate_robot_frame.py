from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from calibrate_cameras import (
    average_transforms,
    intrinsics_dict,
    intrinsics_values,
    next_color,
    rotation_error_deg,
    start_camera,
    transform_from_pose,
)

TOP_CAMERA_SERIAL = "323622271046"
OUTPUT_PATH = Path(__file__).resolve().parent / "calibration" / "base_from_top.json"
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TagDetection:
    marker_id: int
    corners: np.ndarray
    transform: np.ndarray
    reprojection_error: float


@dataclass
class BoardObservation:
    transform: np.ndarray
    corners: list[np.ndarray]
    ids: np.ndarray
    reprojection_error: float


@dataclass
class CalibrationSample:
    base_from_flange: np.ndarray
    top_from_board: np.ndarray
    reprojection_error: float
    visible_tags: int


@dataclass
class RobotCalibrationResult:
    base_from_top: np.ndarray
    flange_from_board: np.ndarray
    scale: float
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    inliers: np.ndarray


class PiperPoseReader:
    def __init__(self, can_interface: str):
        self.can_interface = can_interface
        self.interface: Any = None

    def connect(self) -> None:
        can_path = Path("/sys/class/net") / self.can_interface
        if not can_path.exists():
            raise RuntimeError(
                f"CAN interface {self.can_interface} is not active. Run:\n"
                "python ~/projects/distal/distal/hardware/can_activate.py "
                "-p 3-1:1.0=can_arm_right"
            )
        if (can_path / "operstate").read_text().strip() != "up":
            raise RuntimeError(
                f"CAN interface {self.can_interface} is down. Run:\n"
                "python ~/projects/distal/distal/hardware/can_activate.py "
                "-p 3-1:1.0=can_arm_right"
            )
        module = import_piper_sdk()
        self.interface = module.C_PiperInterface_V2(self.can_interface)
        self.interface.ConnectPort(piper_init=False)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self.read() is not None:
                return
            time.sleep(0.05)
        self.disconnect()
        raise RuntimeError(
            f"No end-pose feedback on {self.can_interface}. "
            "Plug in the arm and start its normal teleoperation process."
        )

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
        return transform_from_rpy(rpy, translation)

    def disconnect(self) -> None:
        if self.interface is not None:
            self.interface.DisconnectPort()
            self.interface = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("right", "left"), default="right")
    parser.add_argument("--can-interface")
    parser.add_argument("--top-serial", default=TOP_CAMERA_SERIAL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_16h5")
    parser.add_argument("--anchor-id", type=int, default=7)
    parser.add_argument("--nominal-tag-mm", type=float, default=50.0)
    parser.add_argument("--map-frames", type=int, default=30)
    parser.add_argument("--map-min-tags", type=int, default=5)
    parser.add_argument("--min-tags", type=int, default=3)
    parser.add_argument("--poses", type=int, default=15)
    parser.add_argument("--min-poses", type=int, default=10)
    parser.add_argument("--stability-ms", type=float, default=650.0)
    parser.add_argument("--max-stability-translation-mm", type=float, default=3.0)
    parser.add_argument("--max-stability-rotation-deg", type=float, default=1.0)
    parser.add_argument("--min-translation-mm", type=float, default=25.0)
    parser.add_argument("--min-rotation-deg", type=float, default=7.0)
    parser.add_argument("--max-reprojection-px", type=float, default=5.0)
    parser.add_argument("--max-translation-residual-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-residual-deg", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


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
            raise RuntimeError(
                "Piper SDK is unavailable. Run `pixi install` in the repository."
            ) from None
        sys.path.append(str(site_packages[-1]))
        return importlib.import_module("piper_sdk")


def destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


def validate_args(args: argparse.Namespace) -> None:
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("Camera resolution and frame rate must be positive")
    if args.nominal_tag_mm <= 0:
        raise ValueError("Nominal tag size must be positive")
    if args.map_frames < 5 or args.map_min_tags < 2:
        raise ValueError("Marker-map capture needs at least 5 frames and 2 tags")
    if args.min_tags < 1 or args.min_tags > args.map_min_tags:
        raise ValueError("min-tags must be between 1 and map-min-tags")
    if not 6 <= args.min_poses <= args.poses:
        raise ValueError("Pose counts must satisfy 6 <= min-poses <= poses")
    if args.stability_ms <= 0:
        raise ValueError("Stability duration must be positive")
    if args.can_interface is None:
        args.can_interface = f"can_arm_{args.arm}"


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


def marker_points(size_m: float) -> np.ndarray:
    half = size_m * 0.5
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def dictionary_from_name(name: str):
    dictionary_id = getattr(cv2.aruco, name, None)
    if dictionary_id is None or not name.startswith("DICT_"):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def create_detector(dictionary):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 63
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.008
    parameters.minCornerDistanceRate = 0.01
    parameters.minDistanceToBorder = 2
    parameters.aprilTagQuadDecimate = 1.0
    parameters.aprilTagQuadSigma = 0.0
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected = cv2.projectPoints(
        object_points,
        rotation,
        translation,
        matrix,
        distortion,
    )[0].reshape(-1, 2)
    return float(
        np.sqrt(np.mean(np.sum((projected - image_points.reshape(-1, 2)) ** 2, axis=1)))
    )


def solve_tag_pose(
    corners: np.ndarray,
    points: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    result = cv2.solvePnPGeneric(
        points,
        corners.reshape(4, 2),
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not result[0]:
        return None
    candidates = []
    for rotation, translation in zip(result[1], result[2], strict=True):
        if float(translation.reshape(3)[2]) <= 0.0:
            continue
        error = reprojection_error(
            points,
            corners,
            rotation,
            translation,
            matrix,
            distortion,
        )
        candidates.append((error, rotation, translation))
    if not candidates:
        return None
    error, rotation, translation = min(candidates, key=lambda candidate: candidate[0])
    rotation, translation = cv2.solvePnPRefineLM(
        points,
        corners.reshape(4, 2),
        matrix,
        distortion,
        rotation,
        translation,
    )
    error = reprojection_error(
        points,
        corners,
        rotation,
        translation,
        matrix,
        distortion,
    )
    return transform_from_pose(rotation, translation), error


def detect_tags(
    image: np.ndarray,
    detector,
    points: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[TagDetection]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    detections = []
    for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True):
        solved = solve_tag_pose(
            marker_corners.reshape(4, 2).astype(np.float32),
            points,
            matrix,
            distortion,
        )
        if solved is None:
            continue
        transform, error = solved
        detections.append(
            TagDetection(
                int(marker_id),
                marker_corners.reshape(4, 2).astype(np.float32),
                transform,
                error,
            )
        )
    return detections


def robust_average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    estimate = average_transforms(transforms)
    if len(transforms) < 5:
        return estimate
    translations = np.asarray(
        [np.linalg.norm(item[:3, 3] - estimate[:3, 3]) for item in transforms]
    )
    rotations = np.asarray(
        [rotation_error_deg(estimate[:3, :3], item[:3, :3]) for item in transforms]
    )
    translation_limit = robust_limit(translations, 0.003)
    rotation_limit = robust_limit(rotations, 1.0)
    selected = [
        item
        for item, translation, rotation in zip(
            transforms,
            translations,
            rotations,
            strict=True,
        )
        if translation <= translation_limit and rotation <= rotation_limit
    ]
    return average_transforms(selected if len(selected) >= 3 else transforms)


def build_marker_map(
    frames: list[list[TagDetection]],
    anchor_id: int,
    points: np.ndarray,
    minimum_tags: int,
) -> dict[int, np.ndarray]:
    observations: dict[int, list[np.ndarray]] = {}
    for detections in frames:
        lookup = {detection.marker_id: detection for detection in detections}
        anchor = lookup.get(anchor_id)
        if anchor is None:
            continue
        anchor_from_camera = np.linalg.inv(anchor.transform)
        for marker_id, detection in lookup.items():
            anchor_from_tag = anchor_from_camera @ detection.transform
            observations.setdefault(marker_id, []).append(anchor_from_tag)
    marker_map = {}
    for marker_id, transforms in observations.items():
        if len(transforms) < max(3, len(frames) // 4):
            continue
        anchor_from_tag = robust_average_transforms(transforms)
        mapped = transform_points(points, anchor_from_tag)
        mapped[:, 2] = 0.0
        marker_map[marker_id] = mapped.astype(np.float32)
    marker_map[anchor_id] = points.copy()
    if len(marker_map) < minimum_tags:
        raise RuntimeError(
            f"Marker map has only {len(marker_map)} tags; need {minimum_tags}"
        )
    return marker_map


def observe_board(
    detections: list[TagDetection],
    marker_map: dict[int, np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
    minimum_tags: int,
) -> BoardObservation | None:
    selected = [
        detection for detection in detections if detection.marker_id in marker_map
    ]
    if len(selected) < minimum_tags:
        return None
    object_points = np.concatenate(
        [marker_map[detection.marker_id] for detection in selected]
    ).astype(np.float32)
    image_points = np.concatenate([detection.corners for detection in selected]).astype(
        np.float32
    )
    success, rotation, translation = cv2.solvePnP(
        object_points,
        image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success or float(translation.reshape(3)[2]) <= 0.0:
        return None
    rotation, translation = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        matrix,
        distortion,
        rotation,
        translation,
    )
    error = reprojection_error(
        object_points,
        image_points,
        rotation,
        translation,
        matrix,
        distortion,
    )
    return BoardObservation(
        transform_from_pose(rotation, translation),
        [detection.corners.reshape(1, 4, 2) for detection in selected],
        np.asarray([detection.marker_id for detection in selected], dtype=np.int32),
        error,
    )


def draw_detection_view(
    image: np.ndarray,
    detections: list[TagDetection],
    observation: BoardObservation | None,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    view = image.copy()
    if detections:
        cv2.aruco.drawDetectedMarkers(
            view,
            [detection.corners.reshape(1, 4, 2) for detection in detections],
            np.asarray(
                [detection.marker_id for detection in detections],
                dtype=np.int32,
            ).reshape(-1, 1),
            (70, 220, 110),
        )
    if observation is not None:
        rotation = cv2.Rodrigues(observation.transform[:3, :3])[0]
        cv2.drawFrameAxes(
            view,
            matrix,
            distortion,
            rotation,
            observation.transform[:3, 3],
            0.05,
            3,
        )
    return view


def draw_status(
    image: np.ndarray,
    headline: str,
    detail: str,
    notice: str,
) -> np.ndarray:
    view = image.copy()
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (view.shape[1], 92), (5, 8, 12), -1)
    cv2.addWeighted(overlay, 0.88, view, 0.12, 0.0, view)
    cv2.putText(
        view,
        headline,
        (18, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (84, 220, 139),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        detail,
        (18, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (218, 224, 232),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        notice,
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 154, 173),
        1,
        cv2.LINE_AA,
    )
    return view


def show_view(window: str, image: np.ndarray) -> int:
    maximum_width = 1400
    maximum_height = 820
    scale = min(
        1.0,
        maximum_width / image.shape[1],
        maximum_height / image.shape[0],
    )
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imshow(window, image)
    return cv2.waitKey(1) & 0xFF


def capture_marker_map(
    pipeline,
    detector,
    points: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    args: argparse.Namespace,
    window: str,
) -> dict[int, np.ndarray]:
    frames: list[list[TagDetection]] = []
    notice = f"Keep tag {args.anchor_id} visible | Q cancels"
    while len(frames) < args.map_frames:
        image = next_color(pipeline)
        detections = detect_tags(image, detector, points, matrix, distortion)
        ids = {detection.marker_id for detection in detections}
        ready = args.anchor_id in ids and len(detections) >= args.map_min_tags
        if ready:
            frames.append(detections)
        detail = (
            f"{len(detections)} tags detected | "
            f"reference frames {len(frames)}/{args.map_frames}"
        )
        if not ready:
            detail += f" | need tag {args.anchor_id} and {args.map_min_tags}+ tags"
        view = draw_detection_view(
            image,
            detections,
            None,
            matrix,
            distortion,
        )
        view = draw_status(view, "LEARNING THE RIGID TAG BOARD", detail, notice)
        key = show_view(window, view)
        if key in (27, ord("q")):
            raise RuntimeError("Calibration cancelled")
    marker_map = build_marker_map(
        frames,
        args.anchor_id,
        points,
        args.map_min_tags,
    )
    print(f"Learned {len(marker_map)} rigid tag positions")
    return marker_map


def sample_is_stable(
    first: CalibrationSample,
    second: CalibrationSample,
    args: argparse.Namespace,
) -> bool:
    maximum_translation = args.max_stability_translation_mm / 1000.0
    maximum_rotation = args.max_stability_rotation_deg
    robot_translation = np.linalg.norm(
        first.base_from_flange[:3, 3] - second.base_from_flange[:3, 3]
    )
    robot_rotation = rotation_error_deg(
        first.base_from_flange[:3, :3],
        second.base_from_flange[:3, :3],
    )
    board_translation = np.linalg.norm(
        first.top_from_board[:3, 3] - second.top_from_board[:3, 3]
    )
    board_rotation = rotation_error_deg(
        first.top_from_board[:3, :3],
        second.top_from_board[:3, :3],
    )
    return (
        robot_translation <= maximum_translation
        and board_translation <= maximum_translation
        and robot_rotation <= maximum_rotation
        and board_rotation <= maximum_rotation
    )


def average_samples(samples: list[CalibrationSample]) -> CalibrationSample:
    return CalibrationSample(
        average_transforms([sample.base_from_flange for sample in samples]),
        average_transforms([sample.top_from_board for sample in samples]),
        float(np.median([sample.reprojection_error for sample in samples])),
        round(float(np.median([sample.visible_tags for sample in samples]))),
    )


def pose_is_diverse(
    candidate: CalibrationSample,
    samples: list[CalibrationSample],
    args: argparse.Namespace,
) -> bool:
    for sample in samples:
        translation = np.linalg.norm(
            candidate.base_from_flange[:3, 3] - sample.base_from_flange[:3, 3]
        )
        rotation = rotation_error_deg(
            candidate.base_from_flange[:3, :3],
            sample.base_from_flange[:3, :3],
        )
        if (
            translation < args.min_translation_mm / 1000.0
            and rotation < args.min_rotation_deg
        ):
            return False
    return True


def capture_samples(
    pipeline,
    pose_reader: PiperPoseReader,
    detector,
    points: np.ndarray,
    marker_map: dict[int, np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
    args: argparse.Namespace,
    window: str,
) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    stable_window: list[tuple[float, CalibrationSample]] = []
    notice = "Move and tilt the gripper, then hold still"
    print("Teleoperate the gripper through varied positions and rotations")
    print("Each useful pose is captured automatically when the arm is still")
    while len(samples) < args.poses:
        image = next_color(pipeline)
        detections = detect_tags(image, detector, points, matrix, distortion)
        observation = observe_board(
            detections,
            marker_map,
            matrix,
            distortion,
            args.min_tags,
        )
        base_from_flange = pose_reader.read()
        current = None
        if (
            observation is not None
            and observation.reprojection_error <= args.max_reprojection_px
            and base_from_flange is not None
        ):
            current = CalibrationSample(
                base_from_flange,
                observation.transform,
                observation.reprojection_error,
                len(observation.ids),
            )
        now = time.monotonic()
        if current is None:
            stable_window.clear()
        elif stable_window and not sample_is_stable(stable_window[0][1], current, args):
            stable_window.clear()
            stable_window.append((now, current))
        else:
            stable_window.append((now, current))
        horizon = max(1.5, 3.0 * args.stability_ms / 1000.0)
        stable_window[:] = [item for item in stable_window if item[0] >= now - horizon]
        progress = 0.0
        if stable_window:
            progress = min(
                1.0,
                (stable_window[-1][0] - stable_window[0][0])
                / (args.stability_ms / 1000.0),
            )
        if progress >= 1.0 and len(stable_window) >= 4:
            candidate = average_samples([item[1] for item in stable_window])
            if pose_is_diverse(candidate, samples, args):
                samples.append(candidate)
                notice = f"Captured pose {len(samples)}; move to a new angle"
                print(notice)
            else:
                notice = "Pose repeated; move or rotate farther"
            stable_window.clear()
            progress = 0.0
        view = draw_detection_view(
            image,
            detections,
            observation,
            matrix,
            distortion,
        )
        state = f"POSES {len(samples)}/{args.poses}"
        if current is None:
            detail = (
                f"Need {args.min_tags}+ mapped tags and live {args.arm}-arm feedback"
            )
        else:
            detail = (
                f"{current.visible_tags} tags | {current.reprojection_error:.2f} px | "
                f"hold still {round(progress * 100)}%"
            )
        view = draw_status(
            view,
            state,
            detail,
            f"{notice} | ENTER finishes after {args.min_poses} | U undo | Q cancel",
        )
        key = show_view(window, view)
        if key in (27, ord("q")):
            raise RuntimeError("Calibration cancelled")
        if key in (10, 13):
            if len(samples) >= args.min_poses:
                break
            notice = f"Need at least {args.min_poses} poses"
        elif key == ord("u"):
            if samples:
                samples.pop()
            stable_window.clear()
            notice = "Removed the latest pose"
    return samples


def robust_limit(values: np.ndarray, floor: float) -> float:
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    return max(floor, median + 3.0 * 1.4826 * deviation)


def solve_translation_and_scale(
    samples: list[CalibrationSample],
    flange_rotation: np.ndarray,
    camera_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    matrix_rows = []
    vector_rows = []
    for sample in samples:
        robot_rotation = sample.base_from_flange[:3, :3]
        robot_translation = sample.base_from_flange[:3, 3]
        board_translation = sample.top_from_board[:3, 3]
        matrix_rows.append(
            np.column_stack(
                (
                    robot_rotation,
                    -np.eye(3),
                    -(camera_rotation @ board_translation).reshape(3, 1),
                )
            )
        )
        vector_rows.append(-robot_translation)
    solution = np.linalg.lstsq(
        np.vstack(matrix_rows),
        np.concatenate(vector_rows),
        rcond=None,
    )[0]
    flange_translation = solution[:3]
    camera_translation = solution[3:6]
    scale = float(solution[6])
    if not np.isfinite(solution).all() or scale <= 0.0:
        raise RuntimeError("Calibration produced an invalid metric scale")
    flange_from_board = np.eye(4, dtype=np.float64)
    flange_from_board[:3, :3] = flange_rotation
    flange_from_board[:3, 3] = flange_translation
    base_from_top = np.eye(4, dtype=np.float64)
    base_from_top[:3, :3] = camera_rotation
    base_from_top[:3, 3] = camera_translation
    return flange_from_board, base_from_top, scale


def solve_once(
    samples: list[CalibrationSample],
) -> tuple[np.ndarray, np.ndarray, float]:
    rotations_robot = [sample.base_from_flange[:3, :3] for sample in samples]
    rotations_board = [sample.top_from_board[:3, :3] for sample in samples]
    equations = []
    for first in range(len(samples) - 1):
        for second in range(first + 1, len(samples)):
            robot_relative = rotations_robot[second].T @ rotations_robot[first]
            board_relative = rotations_board[second].T @ rotations_board[first]
            equations.append(
                np.kron(np.eye(3), robot_relative)
                - np.kron(board_relative.T, np.eye(3))
            )
    _, singular_values, right = np.linalg.svd(np.vstack(equations))
    if singular_values[-2] < 1e-4:
        raise RuntimeError("Robot poses do not contain enough rotational diversity")
    flange_rotation = right[-1].reshape(3, 3, order="F")
    if np.linalg.det(flange_rotation) < 0.0:
        flange_rotation *= -1.0
    left, _, right_rotation = np.linalg.svd(flange_rotation)
    flange_rotation = left @ right_rotation
    if np.linalg.det(flange_rotation) < 0.0:
        left[:, -1] *= -1.0
        flange_rotation = left @ right_rotation
    camera_rotation = average_rotations(
        [
            robot_rotation @ flange_rotation @ board_rotation.T
            for robot_rotation, board_rotation in zip(
                rotations_robot,
                rotations_board,
                strict=True,
            )
        ]
    )
    return solve_translation_and_scale(
        samples,
        flange_rotation,
        camera_rotation,
    )


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    left, _, right = np.linalg.svd(np.sum(rotations, axis=0))
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def calibration_errors(
    samples: list[CalibrationSample],
    flange_from_board: np.ndarray,
    base_from_top: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    translations = []
    rotations = []
    for sample in samples:
        top_from_board = sample.top_from_board.copy()
        top_from_board[:3, 3] *= scale
        robot_side = sample.base_from_flange @ flange_from_board
        camera_side = base_from_top @ top_from_board
        translations.append(np.linalg.norm(robot_side[:3, 3] - camera_side[:3, 3]))
        rotations.append(rotation_error_deg(robot_side[:3, :3], camera_side[:3, :3]))
    return np.asarray(translations), np.asarray(rotations)


def solve_calibration(
    samples: list[CalibrationSample],
    minimum_inliers: int,
    maximum_translation_m: float,
    maximum_rotation_deg: float,
) -> RobotCalibrationResult:
    flange_from_board, base_from_top, scale = solve_once(samples)
    translations, rotations = calibration_errors(
        samples,
        flange_from_board,
        base_from_top,
        scale,
    )
    translation_limit = min(
        maximum_translation_m,
        robust_limit(translations, 0.008),
    )
    rotation_limit = min(
        maximum_rotation_deg,
        robust_limit(rotations, 1.25),
    )
    inliers = (translations <= translation_limit) & (rotations <= rotation_limit)
    if int(inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Only {int(inliers.sum())}/{len(samples)} robot poses agree. "
            "Repeat with wider translations and rotations."
        )
    selected = [sample for sample, keep in zip(samples, inliers, strict=True) if keep]
    flange_from_board, base_from_top, scale = solve_once(selected)
    translations, rotations = calibration_errors(
        samples,
        flange_from_board,
        base_from_top,
        scale,
    )
    inliers = (translations <= maximum_translation_m) & (
        rotations <= maximum_rotation_deg
    )
    if int(inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Only {int(inliers.sum())}/{len(samples)} poses pass final validation"
        )
    return RobotCalibrationResult(
        base_from_top,
        flange_from_board,
        scale,
        translations,
        rotations,
        inliers,
    )


def save_calibration(
    args: argparse.Namespace,
    result: RobotCalibrationResult,
    samples: list[CalibrationSample],
    marker_map: dict[int, np.ndarray],
    top_intrinsics,
) -> None:
    inlier_translations = result.translation_errors_m[result.inliers] * 1000.0
    inlier_rotations = result.rotation_errors_deg[result.inliers]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frame": f"piper_{args.arm}_base_frame",
        "source_frame": "top_camera_optical_frame",
        "base_from_top": result.base_from_top.tolist(),
        "flange_from_board": result.flange_from_board.tolist(),
        "arm": args.arm,
        "can_interface": args.can_interface,
        "top_serial": args.top_serial,
        "resolution": [args.width, args.height],
        "board": {
            "type": "irregular_apriltag_map",
            "dictionary": args.dictionary,
            "anchor_id": args.anchor_id,
            "nominal_tag_mm": args.nominal_tag_mm,
            "estimated_tag_mm": args.nominal_tag_mm * result.scale,
            "metric_scale": result.scale,
            "object_points_nominal_m": {
                str(marker_id): corners.tolist()
                for marker_id, corners in sorted(marker_map.items())
            },
        },
        "samples": len(samples),
        "inliers": int(result.inliers.sum()),
        "translation_residual_mm": {
            "median": float(np.median(inlier_translations)),
            "maximum": float(np.max(inlier_translations)),
        },
        "rotation_residual_deg": {
            "median": float(np.median(inlier_rotations)),
            "maximum": float(np.max(inlier_rotations)),
        },
        "top_intrinsics": intrinsics_dict(top_intrinsics),
        "per_pose": [
            {
                "inlier": bool(result.inliers[index]),
                "translation_residual_mm": float(
                    result.translation_errors_m[index] * 1000.0
                ),
                "rotation_residual_deg": float(result.rotation_errors_deg[index]),
                "reprojection_px": sample.reprojection_error,
                "visible_tags": sample.visible_tags,
            }
            for index, sample in enumerate(samples)
        ],
    }
    encoded = json.dumps(payload, indent=2).encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, args.output)
    print(f"Saved {args.output}")
    print(
        f"translation residual: median {np.median(inlier_translations):.1f} mm, "
        f"max {np.max(inlier_translations):.1f} mm"
    )
    print(
        f"rotation residual: median {np.median(inlier_rotations):.2f} deg, "
        f"max {np.max(inlier_rotations):.2f} deg"
    )
    print(f"estimated printed tag edge: {args.nominal_tag_mm * result.scale:.1f} mm")


def rotation_vector(values: tuple[float, float, float]) -> np.ndarray:
    return cv2.Rodrigues(np.asarray(values, dtype=np.float64))[0]


def synthetic_transform(
    rotation: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_vector(rotation)
    transform[:3, 3] = translation
    return transform


def synthetic_marker_map(points: np.ndarray) -> dict[int, np.ndarray]:
    layout = {
        0: (-0.11, -0.10, -0.24),
        1: (0.0, -0.115, 0.15),
        2: (0.115, -0.095, -0.45),
        3: (-0.125, 0.0, 0.38),
        4: (0.0, 0.0, -0.12),
        5: (0.12, 0.01, 0.48),
        6: (-0.105, 0.105, -0.42),
        7: (0.01, 0.11, 0.0),
        8: (0.12, 0.115, 0.27),
    }
    marker_map = {}
    for marker_id, (x, y, angle) in layout.items():
        transform = synthetic_transform((0.0, 0.0, angle), (x, y, 0.0))
        marker_map[marker_id] = transform_points(points, transform).astype(np.float32)
    return marker_map


def render_marker_map(
    marker_map: dict[int, np.ndarray],
    dictionary,
    top_from_board: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    canvas = np.full((720, 1280), 238, dtype=np.uint8)
    rotation = cv2.Rodrigues(top_from_board[:3, :3])[0]
    for marker_id, object_points in marker_map.items():
        destination = cv2.projectPoints(
            object_points,
            rotation,
            top_from_board[:3, 3],
            matrix,
            distortion,
        )[0].reshape(4, 2)
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 180, borderBits=1)
        patch = np.full((240, 240), 255, dtype=np.uint8)
        patch[30:210, 30:210] = marker
        source = np.asarray(
            [[30.0, 30.0], [209.0, 30.0], [209.0, 209.0], [30.0, 209.0]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, destination.astype(np.float32))
        warped = cv2.warpPerspective(
            patch,
            homography,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=255,
        )
        coverage = cv2.warpPerspective(
            np.full(patch.shape, 255, dtype=np.uint8),
            homography,
            (canvas.shape[1], canvas.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        canvas[coverage > 0] = warped[coverage > 0]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def run_self_test(args: argparse.Namespace) -> None:
    dictionary = dictionary_from_name(args.dictionary)
    detector = create_detector(dictionary)
    nominal_size = args.nominal_tag_mm / 1000.0
    points = marker_points(nominal_size)
    expected_map = synthetic_marker_map(points)
    matrix = np.asarray(
        [[920.0, 0.0, 640.0], [0.0, 918.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    frames = []
    for index in range(8):
        board_pose = synthetic_transform(
            (
                3.02 + 0.008 * index,
                -0.13 + 0.008 * index,
                0.025 - 0.005 * index,
            ),
            (-0.01 + 0.002 * index, -0.005, 0.72 + 0.004 * index),
        )
        image = render_marker_map(
            expected_map,
            dictionary,
            board_pose,
            matrix,
            distortion,
        )
        detections = detect_tags(image, detector, points, matrix, distortion)
        if len(detections) < 7:
            raise RuntimeError(
                f"Synthetic tag detection found only {len(detections)} tags"
            )
        frames.append(detections)
    learned_map = build_marker_map(frames, args.anchor_id, points, 5)
    check_pose = synthetic_transform((2.96, -0.20, 0.07), (0.015, 0.01, 0.75))
    check_image = render_marker_map(
        expected_map,
        dictionary,
        check_pose,
        matrix,
        distortion,
    )
    observation = observe_board(
        detect_tags(check_image, detector, points, matrix, distortion),
        learned_map,
        matrix,
        distortion,
        3,
    )
    if observation is None or observation.reprojection_error > 2.0:
        raise RuntimeError("Synthetic mapped-board pose estimation failed")
    true_scale = 0.76
    expected_flange_from_board = synthetic_transform(
        (0.08, -0.05, 0.03),
        (0.015, 0.0, 0.095),
    )
    expected_base_from_top = synthetic_transform(
        (-0.14, 0.26, 0.08),
        (0.42, -0.18, 0.72),
    )
    samples = []
    for index in range(18):
        base_from_flange = synthetic_transform(
            (
                -0.45 + 0.07 * (index % 6),
                0.20 - 0.06 * (index % 5),
                -0.35 + 0.09 * (index % 7),
            ),
            (
                0.24 + 0.035 * (index % 5),
                -0.18 + 0.04 * (index % 4),
                0.24 + 0.025 * (index % 6),
            ),
        )
        top_from_board = (
            np.linalg.inv(expected_base_from_top)
            @ base_from_flange
            @ expected_flange_from_board
        )
        top_from_board[:3, 3] /= true_scale
        samples.append(CalibrationSample(base_from_flange, top_from_board, 0.4, 8))
    result = solve_calibration(samples, 10, 0.02, 3.0)
    translation_error = np.linalg.norm(
        result.base_from_top[:3, 3] - expected_base_from_top[:3, 3]
    )
    rotation_error = rotation_error_deg(
        result.base_from_top[:3, :3],
        expected_base_from_top[:3, :3],
    )
    if translation_error > 1e-6 or rotation_error > 1e-5:
        raise RuntimeError("Synthetic robot/camera transform recovery failed")
    if abs(result.scale - true_scale) > 1e-6:
        raise RuntimeError("Synthetic metric-scale recovery failed")
    with tempfile.TemporaryDirectory() as directory:
        dry_args = argparse.Namespace(**vars(args))
        dry_args.output = Path(directory) / "base_from_top.json"
        dry_args.can_interface = dry_args.can_interface or f"can_arm_{args.arm}"
        fake_intrinsics = type(
            "Intrinsics",
            (),
            {
                "width": 1280,
                "height": 720,
                "fx": 920.0,
                "fy": 918.0,
                "ppx": 640.0,
                "ppy": 360.0,
                "model": "brown_conrady",
                "coeffs": [0.0] * 5,
            },
        )()
        save_calibration(
            dry_args,
            result,
            samples,
            learned_map,
            fake_intrinsics,
        )
        saved = json.loads(dry_args.output.read_text())
        if saved["board"]["dictionary"] != "DICT_APRILTAG_16h5":
            raise RuntimeError("Saved calibration metadata is invalid")
    print(
        "Self-test passed: high-resolution tag detection, arbitrary board mapping, "
        "automatic-pose solver, metric scale, and output file"
    )


def run_calibration(args: argparse.Namespace) -> None:
    rs: Any = importlib.import_module("pyrealsense2")
    dictionary = dictionary_from_name(args.dictionary)
    detector = create_detector(dictionary)
    points = marker_points(args.nominal_tag_mm / 1000.0)
    pipeline = None
    pose_reader = PiperPoseReader(args.can_interface)
    window = f"Robot frame calibration | top camera + {args.arm} arm"
    try:
        pipeline, top_intrinsics = start_camera(rs, args.top_serial, args)
        top_matrix, top_distortion = intrinsics_values(top_intrinsics)
        pose_reader.connect()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        marker_map = capture_marker_map(
            pipeline,
            detector,
            points,
            top_matrix,
            top_distortion,
            args,
            window,
        )
        samples = capture_samples(
            pipeline,
            pose_reader,
            detector,
            points,
            marker_map,
            top_matrix,
            top_distortion,
            args,
            window,
        )
        result = solve_calibration(
            samples,
            args.min_poses,
            args.max_translation_residual_mm / 1000.0,
            args.max_rotation_residual_deg,
        )
        save_calibration(args, result, samples, marker_map, top_intrinsics)
    except Exception as error:
        if pipeline is None:
            raise RuntimeError(
                "Could not open the top camera. Stop track_obstacle.sh before "
                f"calibration. Camera error: {error}"
            ) from error
        raise
    finally:
        destroy_windows()
        pose_reader.disconnect()
        if pipeline is not None:
            pipeline.stop()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.self_test:
        run_self_test(args)
        return
    run_calibration(args)


if __name__ == "__main__":
    main()
