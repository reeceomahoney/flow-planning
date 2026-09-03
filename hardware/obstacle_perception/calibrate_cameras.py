from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

TOP_CAMERA_SERIAL = "323622271046"
LEFT_CAMERA_SERIAL = "335122272969"
OUTPUT_PATH = Path(__file__).resolve().parent / "calibration" / "top_from_left.json"


@dataclass
class BoardObservation:
    transform: np.ndarray
    corners: list[np.ndarray]
    ids: np.ndarray
    reprojection_error: float


@dataclass
class PosePair:
    top: BoardObservation
    left: BoardObservation


@dataclass
class CalibrationResult:
    transform: np.ndarray
    translation_errors_m: np.ndarray
    rotation_errors_deg: np.ndarray
    inliers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-serial", default=TOP_CAMERA_SERIAL)
    parser.add_argument("--left-serial", default=LEFT_CAMERA_SERIAL)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--dictionary", default="DICT_APRILTAG_36h11")
    parser.add_argument("--markers-x", type=int, default=6)
    parser.add_argument("--markers-y", type=int, default=6)
    parser.add_argument("--marker-mm", type=float, default=50.0)
    parser.add_argument("--spacing-mm", type=float, default=15.0)
    parser.add_argument("--poses", type=int, default=12)
    parser.add_argument("--min-poses", type=int, default=6)
    parser.add_argument("--min-tags", type=int, default=3)
    parser.add_argument("--min-translation-mm", type=float, default=15.0)
    parser.add_argument("--min-rotation-deg", type=float, default=4.0)
    parser.add_argument("--max-reprojection-px", type=float, default=7.0)
    parser.add_argument("--stability-ms", type=float, default=350.0)
    parser.add_argument("--max-stability-translation-mm", type=float, default=15.0)
    parser.add_argument("--max-stability-rotation-deg", type=float, default=3.0)
    parser.add_argument("--max-translation-residual-mm", type=float, default=20.0)
    parser.add_argument("--max-rotation-residual-deg", type=float, default=2.5)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.markers_x < 2 or args.markers_y < 2:
        raise ValueError("AprilGrid must have at least 2x2 markers")
    if args.marker_mm <= 0 or args.spacing_mm <= 0:
        raise ValueError("Marker size and spacing must be positive")
    maximum_tags = args.markers_x * args.markers_y
    if not 2 <= args.min_tags <= maximum_tags:
        raise ValueError(f"Minimum tags must be between 2 and {maximum_tags}")
    if not 3 <= args.min_poses <= args.poses:
        raise ValueError("Pose counts must satisfy 3 <= min-poses <= poses")
    if args.stability_ms <= 0:
        raise ValueError("Stability duration must be positive")
    if args.max_stability_translation_mm <= 0:
        raise ValueError("Stability translation must be positive")
    if args.max_stability_rotation_deg <= 0:
        raise ValueError("Stability rotation must be positive")


def create_board(args: argparse.Namespace):
    dictionary_id = getattr(cv2.aruco, args.dictionary, None)
    if dictionary_id is None or not args.dictionary.startswith("DICT_"):
        raise ValueError(f"Unknown ArUco dictionary: {args.dictionary}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    marker_m = args.marker_mm / 1000.0
    step_m = (args.marker_mm + args.spacing_mm) / 1000.0
    object_points = []
    for row in range(args.markers_y):
        for column in range(args.markers_x):
            x = column * step_m
            y = (args.markers_y - 1 - row) * step_m
            object_points.append(
                np.asarray(
                    [
                        [x + marker_m, y + marker_m, 0.0],
                        [x, y + marker_m, 0.0],
                        [x, y, 0.0],
                        [x + marker_m, y, 0.0],
                    ],
                    dtype=np.float32,
                )
            )
    return cv2.aruco.Board(
        object_points,
        dictionary,
        np.arange(args.markers_x * args.markers_y, dtype=np.int32),
    )


def intrinsics_values(intrinsics) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.asarray(intrinsics.coeffs, dtype=np.float64)
    return matrix, distortion


def create_detector(board, matrix: np.ndarray, distortion: np.ndarray):
    del matrix, distortion
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minCornerDistanceRate = 0.01
    return cv2.aruco.ArucoDetector(board.getDictionary(), parameters)


def transform_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = cv2.Rodrigues(rotation)[0]
    transform[:3, 3] = translation.reshape(3)
    return transform


def match_grid_points(
    board, corners: list[np.ndarray], ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {
        int(marker_id): np.asarray(points).reshape(4, 3)
        for points, marker_id in zip(
            board.getObjPoints(),
            board.getIds().reshape(-1),
            strict=True,
        )
    }
    try:
        object_points = np.concatenate(
            [lookup[int(marker_id)] for marker_id in ids.reshape(-1)]
        ).astype(np.float32)
    except KeyError as error:
        raise RuntimeError(f"Unexpected tag ID: {error.args[0]}") from error
    image_points = np.concatenate([corner.reshape(4, 2) for corner in corners]).astype(
        np.float32
    )
    return object_points, image_points


def observe_board(
    image: np.ndarray,
    board,
    detector,
    matrix: np.ndarray,
    distortion: np.ndarray,
    min_tags: int,
) -> BoardObservation | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is not None:
        corners, ids, _, _ = detector.refineDetectedMarkers(
            gray,
            board,
            corners,
            ids,
            rejected,
            matrix,
            distortion,
        )
    if ids is None or len(corners) < min_tags:
        return None
    object_points, image_points = match_grid_points(board, corners, ids)
    success, rotation, translation = cv2.solvePnP(
        object_points,
        image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not success or float(translation.reshape(3)[2]) <= 0:
        return None
    rotation, translation = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        matrix,
        distortion,
        rotation,
        translation,
    )
    projected = cv2.projectPoints(
        object_points,
        rotation,
        translation,
        matrix,
        distortion,
    )[0].reshape(-1, 2)
    error = float(
        np.sqrt(np.mean(np.sum((projected - image_points.reshape(-1, 2)) ** 2, axis=1)))
    )
    return BoardObservation(
        transform_from_pose(rotation, translation),
        corners,
        ids,
        error,
    )


def rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    translation = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    rotation_sum = np.sum([transform[:3, :3] for transform in transforms], axis=0)
    left, _, right = np.linalg.svd(rotation_sum)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    average = np.eye(4, dtype=np.float64)
    average[:3, :3] = rotation
    average[:3, 3] = translation
    return average


def transform_errors(
    transforms: list[np.ndarray], estimate: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    translations = np.asarray(
        [np.linalg.norm(transform[:3, 3] - estimate[:3, 3]) for transform in transforms]
    )
    rotations = np.asarray(
        [
            rotation_error_deg(estimate[:3, :3], transform[:3, :3])
            for transform in transforms
        ]
    )
    return translations, rotations


def robust_limit(values: np.ndarray, floor: float) -> float:
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median)))
    return max(floor, median + 3.0 * 1.4826 * deviation)


def solve_extrinsics(
    samples: list[PosePair], minimum_inliers: int
) -> CalibrationResult:
    transforms = [
        pair.top.transform @ np.linalg.inv(pair.left.transform) for pair in samples
    ]
    initial = average_transforms(transforms)
    translations, rotations = transform_errors(transforms, initial)
    translation_limit = robust_limit(translations, 0.012)
    rotation_limit = robust_limit(rotations, 1.5)
    inliers = (translations <= translation_limit) & (rotations <= rotation_limit)
    if int(inliers.sum()) < minimum_inliers:
        raise RuntimeError(
            f"Only {int(inliers.sum())}/{len(samples)} calibration poses agree"
        )
    estimate = average_transforms(
        [transform for transform, keep in zip(transforms, inliers, strict=True) if keep]
    )
    translations, rotations = transform_errors(transforms, estimate)
    return CalibrationResult(estimate, translations, rotations, inliers)


def pose_is_diverse(
    candidate: PosePair,
    samples: list[PosePair],
    minimum_translation_m: float,
    minimum_rotation_deg: float,
) -> bool:
    for sample in samples:
        translation = np.linalg.norm(
            candidate.top.transform[:3, 3] - sample.top.transform[:3, 3]
        )
        rotation = rotation_error_deg(
            candidate.top.transform[:3, :3],
            sample.top.transform[:3, :3],
        )
        if translation < minimum_translation_m and rotation < minimum_rotation_deg:
            return False
    return True


def accept_candidate(
    candidate: PosePair | None,
    samples: list[PosePair],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if candidate is None:
        return False, "Not captured: board must be detected in both views"
    if (
        max(
            candidate.top.reprojection_error,
            candidate.left.reprojection_error,
        )
        > args.max_reprojection_px
    ):
        return False, "Not captured: reprojection error is too high"
    if not pose_is_diverse(
        candidate,
        samples,
        args.min_translation_mm / 1000.0,
        args.min_rotation_deg,
    ):
        return False, "Not captured: move or tilt the board farther"
    samples.append(candidate)
    return True, f"Captured pose {len(samples)}; move and tilt the board"


def observations_are_stable(
    first: BoardObservation,
    second: BoardObservation,
    args: argparse.Namespace,
) -> bool:
    translation = np.linalg.norm(first.transform[:3, 3] - second.transform[:3, 3])
    rotation = rotation_error_deg(
        first.transform[:3, :3],
        second.transform[:3, :3],
    )
    return (
        translation <= args.max_stability_translation_mm / 1000.0
        and rotation <= args.max_stability_rotation_deg
    )


def update_observation_window(
    window: list[tuple[float, BoardObservation]],
    observation: BoardObservation | None,
    timestamp: float,
    args: argparse.Namespace,
) -> None:
    horizon = max(1.0, 3.0 * args.stability_ms / 1000.0)
    window[:] = [(seen, item) for seen, item in window if seen >= timestamp - horizon]
    if observation is None or observation.reprojection_error > args.max_reprojection_px:
        return
    if window and not observations_are_stable(window[0][1], observation, args):
        window.clear()
    window.append((timestamp, observation))


def average_observations(observations: list[BoardObservation]) -> BoardObservation:
    best = max(
        observations,
        key=lambda observation: (
            len(observation.corners),
            -observation.reprojection_error,
        ),
    )
    return BoardObservation(
        average_transforms([observation.transform for observation in observations]),
        best.corners,
        best.ids,
        float(np.median([item.reprojection_error for item in observations])),
    )


def stable_candidate(
    top_window: list[tuple[float, BoardObservation]],
    left_window: list[tuple[float, BoardObservation]],
    args: argparse.Namespace,
) -> tuple[PosePair | None, float]:
    if len(top_window) < 3 or len(left_window) < 3:
        return None, 0.0
    overlap_start = max(top_window[0][0], left_window[0][0])
    overlap_end = min(top_window[-1][0], left_window[-1][0])
    required = args.stability_ms / 1000.0
    progress = float(np.clip((overlap_end - overlap_start) / required, 0.0, 1.0))
    if progress < 1.0:
        return None, progress
    top_observations = [item for seen, item in top_window if seen >= overlap_start]
    left_observations = [item for seen, item in left_window if seen >= overlap_start]
    if len(top_observations) < 3 or len(left_observations) < 3:
        return None, progress
    return (
        PosePair(
            average_observations(top_observations),
            average_observations(left_observations),
        ),
        progress,
    )


def draw_observation(
    image: np.ndarray,
    observation: BoardObservation | None,
    matrix: np.ndarray,
    distortion: np.ndarray,
    name: str,
) -> np.ndarray:
    view = image.copy()
    if observation is None:
        color = (70, 90, 240)
        detail = "board not found"
    else:
        color = (70, 220, 110)
        detail = (
            f"{len(observation.corners)} tags | {observation.reprojection_error:.2f} px"
        )
        cv2.aruco.drawDetectedMarkers(
            view,
            observation.corners,
            observation.ids.reshape(-1, 1),
            color,
        )
        rotation = cv2.Rodrigues(observation.transform[:3, :3])[0]
        translation = observation.transform[:3, 3]
        cv2.drawFrameAxes(
            view,
            matrix,
            distortion,
            rotation,
            translation,
            0.04,
            2,
        )
    cv2.putText(
        view,
        name.upper(),
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        detail,
        (14, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )
    return view


def overlay_instructions(
    combined: np.ndarray,
    count: int,
    target: int,
    notice: str,
    top_ready: bool,
    left_ready: bool,
    stability_progress: float,
) -> None:
    height = 74
    overlay = combined.copy()
    cv2.rectangle(
        overlay,
        (0, combined.shape[0] - height),
        (combined.shape[1], combined.shape[0]),
        (5, 8, 12),
        -1,
    )
    cv2.addWeighted(overlay, 0.88, combined, 0.12, 0.0, combined)
    state = f"POSES {count}/{target}"
    if top_ready and left_ready:
        percent = round(100.0 * stability_progress)
        action = f"HOLD STILL {percent}% | capture is automatic"
    elif top_ready:
        action = "TOP READY | rotate board toward left camera"
    elif left_ready:
        action = "LEFT READY | rotate board toward top camera"
    else:
        action = "Rotate board until both cameras detect it"
    cv2.putText(
        combined,
        state,
        (14, combined.shape[0] - 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (84, 220, 139),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        action,
        (170, combined.shape[0] - 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 218, 229),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        combined,
        f"{notice} | SPACE force | ENTER finish | U undo | R reset | Q cancel",
        (14, combined.shape[0] - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (123, 137, 157),
        1,
        cv2.LINE_AA,
    )


def start_camera(rs, serial: str, args: argparse.Namespace):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(
        rs.stream.color,
        args.width,
        args.height,
        rs.format.bgr8,
        args.fps,
    )
    profile = pipeline.start(config)
    intrinsics = (
        profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
    )
    for _ in range(20):
        pipeline.wait_for_frames(3000)
    return pipeline, intrinsics


def next_color(pipeline) -> np.ndarray:
    frame = pipeline.wait_for_frames(3000).get_color_frame()
    if not frame:
        raise RuntimeError("Camera did not return a color frame")
    return np.asanyarray(frame.get_data()).copy()


def capture_samples(args: argparse.Namespace, board) -> tuple[list[PosePair], Any, Any]:
    rs: Any = importlib.import_module("pyrealsense2")
    top_pipeline = None
    left_pipeline = None
    try:
        top_pipeline, top_intrinsics = start_camera(rs, args.top_serial, args)
        left_pipeline, left_intrinsics = start_camera(rs, args.left_serial, args)
    except Exception as error:
        if top_pipeline is not None:
            top_pipeline.stop()
        raise RuntimeError(
            "Could not open both cameras. Stop track_obstacle.sh before calibration. "
            f"Camera error: {error}"
        ) from error
    top_matrix, top_distortion = intrinsics_values(top_intrinsics)
    left_matrix, left_distortion = intrinsics_values(left_intrinsics)
    top_detector = create_detector(board, top_matrix, top_distortion)
    left_detector = create_detector(board, left_matrix, left_distortion)
    samples: list[PosePair] = []
    top_window: list[tuple[float, BoardObservation]] = []
    left_window: list[tuple[float, BoardObservation]] = []
    notice = "Show the board to both cameras"
    window = "Static camera calibration | top and left wrist"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    display_scale = min(1.0, 1800.0 / (2 * args.width), 760.0 / args.height)
    display_size = (
        round(2 * args.width * display_scale),
        round(args.height * display_scale),
    )
    cv2.resizeWindow(window, *display_size)
    print("Show the AprilGrid to both cameras")
    print("Slowly move and tilt the board; good poses are captured automatically")
    try:
        while True:
            top_image = next_color(top_pipeline)
            left_image = next_color(left_pipeline)
            top_observation = observe_board(
                top_image,
                board,
                top_detector,
                top_matrix,
                top_distortion,
                args.min_tags,
            )
            left_observation = observe_board(
                left_image,
                board,
                left_detector,
                left_matrix,
                left_distortion,
                args.min_tags,
            )
            timestamp = time.monotonic()
            update_observation_window(
                top_window,
                top_observation,
                timestamp,
                args,
            )
            update_observation_window(
                left_window,
                left_observation,
                timestamp,
                args,
            )
            automatic_candidate, stability_progress = stable_candidate(
                top_window,
                left_window,
                args,
            )
            current_candidate = (
                PosePair(top_observation, left_observation)
                if top_observation is not None and left_observation is not None
                else None
            )
            if automatic_candidate is not None:
                captured, notice = accept_candidate(
                    automatic_candidate,
                    samples,
                    args,
                )
                top_window.clear()
                left_window.clear()
                stability_progress = 0.0
                if captured:
                    print(notice)
                if captured and len(samples) >= args.poses:
                    break
            top_view = draw_observation(
                top_image,
                top_observation,
                top_matrix,
                top_distortion,
                "top",
            )
            left_view = draw_observation(
                left_image,
                left_observation,
                left_matrix,
                left_distortion,
                "left wrist",
            )
            combined = np.hstack((top_view, left_view))
            overlay_instructions(
                combined,
                len(samples),
                args.poses,
                notice,
                top_observation is not None,
                left_observation is not None,
                stability_progress,
            )
            display = cv2.resize(
                combined,
                display_size,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(window, display)
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                captured, notice = accept_candidate(current_candidate, samples, args)
                top_window.clear()
                left_window.clear()
                if captured:
                    print(notice)
                if captured and len(samples) >= args.poses:
                    break
            elif key in (10, 13):
                if len(samples) >= args.min_poses:
                    break
                notice = f"Need at least {args.min_poses} poses"
            elif key == ord("u"):
                if samples:
                    samples.pop()
                top_window.clear()
                left_window.clear()
                notice = "Removed the latest pose"
            elif key == ord("r"):
                samples.clear()
                top_window.clear()
                left_window.clear()
                notice = "Cleared all captured poses"
            elif key in (27, ord("q")):
                raise RuntimeError("Calibration cancelled")
    finally:
        cv2.destroyWindow(window)
        top_pipeline.stop()
        left_pipeline.stop()
    return samples, top_intrinsics, left_intrinsics


def intrinsics_dict(intrinsics) -> dict[str, object]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "model": str(intrinsics.model).split(".")[-1],
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def save_calibration(
    args: argparse.Namespace,
    result: CalibrationResult,
    samples: list[PosePair],
    top_intrinsics,
    left_intrinsics,
) -> None:
    inlier_translations = result.translation_errors_m[result.inliers] * 1000.0
    inlier_rotations = result.rotation_errors_deg[result.inliers]
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_frame": "top_camera_optical_frame",
        "source_frame": "left_wrist_camera_optical_frame",
        "top_serial": args.top_serial,
        "left_serial": args.left_serial,
        "resolution": [args.width, args.height],
        "board": {
            "type": "aprilgrid",
            "layout": "kalibr",
            "dictionary": args.dictionary,
            "markers_x": args.markers_x,
            "markers_y": args.markers_y,
            "marker_mm": args.marker_mm,
            "spacing_mm": args.spacing_mm,
        },
        "top_from_left": result.transform.tolist(),
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
        "left_intrinsics": intrinsics_dict(left_intrinsics),
        "per_pose": [
            {
                "inlier": bool(result.inliers[index]),
                "translation_residual_mm": float(
                    result.translation_errors_m[index] * 1000.0
                ),
                "rotation_residual_deg": float(result.rotation_errors_deg[index]),
                "top_reprojection_px": sample.top.reprojection_error,
                "left_reprojection_px": sample.left.reprojection_error,
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


def synthetic_transform(rotation: list[float], translation: list[float]) -> np.ndarray:
    return transform_from_pose(
        np.asarray(rotation, dtype=np.float64),
        np.asarray(translation, dtype=np.float64),
    )


def synthetic_grid_image(args: argparse.Namespace, board) -> np.ndarray:
    marker_pixels = 220
    spacing_pixels = round(marker_pixels * args.spacing_mm / args.marker_mm)
    width = args.markers_x * marker_pixels + (args.markers_x - 1) * spacing_pixels
    height = args.markers_y * marker_pixels + (args.markers_y - 1) * spacing_pixels
    image = np.full((height, width), 255, dtype=np.uint8)
    dictionary = board.getDictionary()
    for row in range(args.markers_y):
        for column in range(args.markers_x):
            marker_id = row * args.markers_x + column
            marker = cv2.aruco.generateImageMarker(
                dictionary,
                marker_id,
                marker_pixels,
                borderBits=1,
            )
            marker = cv2.rotate(marker, cv2.ROTATE_180)
            image_row = args.markers_y - 1 - row
            x = column * (marker_pixels + spacing_pixels)
            y = image_row * (marker_pixels + spacing_pixels)
            image[y : y + marker_pixels, x : x + marker_pixels] = marker
    return image


def render_synthetic_board(
    board_image: np.ndarray,
    transform: np.ndarray,
    matrix: np.ndarray,
    distortion: np.ndarray,
    board_width_m: float,
    board_height_m: float,
) -> np.ndarray:
    object_corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [board_width_m, 0.0, 0.0],
            [board_width_m, board_height_m, 0.0],
            [0.0, board_height_m, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = cv2.Rodrigues(transform[:3, :3])[0]
    projected = cv2.projectPoints(
        object_corners,
        rotation,
        transform[:3, 3],
        matrix,
        distortion,
    )[0].reshape(4, 2)
    source = np.asarray(
        [
            [0.0, 0.0],
            [board_image.shape[1] - 1.0, 0.0],
            [board_image.shape[1] - 1.0, board_image.shape[0] - 1.0],
            [0.0, board_image.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
    gray = cv2.warpPerspective(
        board_image,
        homography,
        (640, 480),
        borderValue=255,
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_self_test(args: argparse.Namespace, board) -> None:
    dry_args = argparse.Namespace(**vars(args))
    dry_args.width = 640
    dry_args.height = 480
    dry_args.fps = 30
    dry_args.min_translation_mm = 1.0
    dry_args.min_rotation_deg = 0.1
    dry_args.stability_ms = 300.0
    matrix = np.asarray(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.zeros(5, dtype=np.float64)
    detector = create_detector(board, matrix, distortion)
    board_image = synthetic_grid_image(args, board)
    board_width_m = (
        args.markers_x * args.marker_mm + (args.markers_x - 1) * args.spacing_mm
    ) / 1000.0
    board_height_m = (
        args.markers_y * args.marker_mm + (args.markers_y - 1) * args.spacing_mm
    ) / 1000.0
    expected = synthetic_transform([0.02, 0.13, -0.025], [0.075, 0.005, 0.018])
    samples = []
    top_window: list[tuple[float, BoardObservation]] = []
    left_window: list[tuple[float, BoardObservation]] = []
    for index in range(12):
        top_board = synthetic_transform(
            [
                0.08 + 0.045 * (index % 3),
                -0.16 + 0.035 * (index % 4),
                -0.04 + 0.018 * (index % 5),
            ],
            [
                -0.075 + 0.018 * (index % 4),
                -0.09 + 0.018 * (index // 4),
                0.54 + 0.018 * (index % 3),
            ],
        )
        left_board = np.linalg.inv(expected) @ top_board
        top_image = render_synthetic_board(
            board_image,
            top_board,
            matrix,
            distortion,
            board_width_m,
            board_height_m,
        )
        left_image = render_synthetic_board(
            board_image,
            left_board,
            matrix,
            distortion,
            board_width_m,
            board_height_m,
        )
        top_observation = observe_board(
            top_image,
            board,
            detector,
            matrix,
            distortion,
            args.min_tags,
        )
        left_observation = observe_board(
            left_image,
            board,
            detector,
            matrix,
            distortion,
            args.min_tags,
        )
        if top_observation is None or left_observation is None:
            raise RuntimeError(f"Synthetic board detection failed at pose {index}")
        if index == 0:
            top_view = draw_observation(
                top_image,
                top_observation,
                matrix,
                distortion,
                "top",
            )
            left_view = draw_observation(
                left_image,
                left_observation,
                matrix,
                distortion,
                "left wrist",
            )
            combined = np.hstack((top_view, left_view))
            overlay_instructions(
                combined,
                0,
                args.poses,
                "dry run",
                True,
                True,
                1.0,
            )
        candidate = None
        for frame_index in range(5):
            timestamp = index * 2.0 + frame_index * 0.1
            update_observation_window(
                top_window,
                top_observation,
                timestamp,
                dry_args,
            )
            update_observation_window(
                left_window,
                left_observation,
                timestamp,
                dry_args,
            )
            candidate, _ = stable_candidate(top_window, left_window, dry_args)
        if candidate is None:
            raise RuntimeError(f"Synthetic automatic capture failed at pose {index}")
        captured, message = accept_candidate(candidate, samples, dry_args)
        if not captured:
            raise RuntimeError(f"Synthetic automatic capture failed: {message}")
        top_window.clear()
        left_window.clear()
        if index == 0:
            duplicate_captured, _ = accept_candidate(candidate, samples, dry_args)
            if duplicate_captured:
                raise RuntimeError("Synthetic duplicate pose was accepted")
    bad_left = BoardObservation(
        samples[-1].left.transform.copy(),
        samples[-1].left.corners,
        samples[-1].left.ids,
        samples[-1].left.reprojection_error,
    )
    bad_left.transform[:3, 3] += np.asarray((0.18, -0.12, 0.09))
    samples.append(PosePair(samples[-1].top, bad_left))
    result = solve_extrinsics(samples, 6)
    translation_error = np.linalg.norm(result.transform[:3, 3] - expected[:3, 3])
    rotation_error = rotation_error_deg(result.transform[:3, :3], expected[:3, :3])
    if translation_error > 0.008 or rotation_error > 1.0:
        raise RuntimeError(
            f"Synthetic calibration error is too high: {translation_error * 1000:.1f} "
            f"mm, {rotation_error:.2f} deg"
        )
    if int(result.inliers.sum()) != 12:
        raise RuntimeError(
            "Synthetic outlier rejection did not retain twelve good poses"
        )
    fake_intrinsics = SimpleNamespace(
        width=640,
        height=480,
        fx=700.0,
        fy=700.0,
        ppx=320.0,
        ppy=240.0,
        model="none",
        coeffs=[0.0] * 5,
    )
    with tempfile.TemporaryDirectory(prefix="camera-calibration-test-") as directory:
        dry_args.output = Path(directory) / "top_from_left.json"
        save_calibration(
            dry_args,
            result,
            samples,
            fake_intrinsics,
            fake_intrinsics,
        )
        saved = json.loads(dry_args.output.read_text())
        saved_transform = np.asarray(saved["top_from_left"], dtype=np.float64)
        if not np.allclose(saved_transform, result.transform):
            raise RuntimeError("Synthetic saved transform does not match solver output")
        if saved["board"]["type"] != "aprilgrid" or saved["inliers"] != 12:
            raise RuntimeError("Synthetic calibration output metadata is invalid")
    print(
        f"self-test passed: {translation_error * 1000:.1f} mm translation, "
        f"{rotation_error:.2f} deg rotation"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    board = create_board(args)
    print(
        f"board: {args.dictionary}, {args.markers_x}x{args.markers_y} tags, "
        f"marker {args.marker_mm:g} mm, spacing {args.spacing_mm:g} mm"
    )
    if args.self_test:
        run_self_test(args, board)
        return
    samples, top_intrinsics, left_intrinsics = capture_samples(args, board)
    result = solve_extrinsics(samples, args.min_poses)
    inlier_translation = result.translation_errors_m[result.inliers] * 1000.0
    inlier_rotation = result.rotation_errors_deg[result.inliers]
    if float(np.max(inlier_translation)) > args.max_translation_residual_mm:
        raise RuntimeError("Translation residual is too high; repeat calibration")
    if float(np.max(inlier_rotation)) > args.max_rotation_residual_deg:
        raise RuntimeError("Rotation residual is too high; repeat calibration")
    save_calibration(
        args,
        result,
        samples,
        top_intrinsics,
        left_intrinsics,
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
