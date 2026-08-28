from __future__ import annotations

import argparse
import importlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TOP_CAMERA_SERIAL = "323622271046"
OUTPUT_PATH = Path(__file__).resolve().parent / "calibration" / "top_roi.json"


@dataclass(frozen=True)
class CropRegion:
    source_width: int
    source_height: int
    source_fps: int
    x: int
    y: int
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default=TOP_CAMERA_SERIAL)
    parser.add_argument("--source-width", type=int, default=1280)
    parser.add_argument("--source-height", type=int, default=720)
    parser.add_argument("--source-fps", type=int, default=5)
    parser.add_argument("--crop-width", type=int, default=640)
    parser.add_argument("--crop-height", type=int, default=480)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if (
        min(
            args.source_width,
            args.source_height,
            args.source_fps,
            args.crop_width,
            args.crop_height,
        )
        <= 0
    ):
        raise ValueError("ROI dimensions and FPS must be positive")
    if args.crop_width > args.source_width or args.crop_height > args.source_height:
        raise ValueError("ROI is larger than the source frame")


def validate_region(region: CropRegion) -> None:
    if (
        min(
            region.source_width,
            region.source_height,
            region.source_fps,
            region.width,
            region.height,
        )
        <= 0
    ):
        raise ValueError("ROI dimensions and FPS must be positive")
    if region.width > region.source_width or region.height > region.source_height:
        raise ValueError("ROI is larger than the source frame")
    if region.x < 0 or region.y < 0:
        raise ValueError("ROI origin must be non-negative")
    if region.x + region.width > region.source_width:
        raise ValueError("ROI extends past the source width")
    if region.y + region.height > region.source_height:
        raise ValueError("ROI extends past the source height")


def region_from_center(center: tuple[int, int], args: argparse.Namespace) -> CropRegion:
    maximum_x = args.source_width - args.crop_width
    maximum_y = args.source_height - args.crop_height
    x = int(np.clip(center[0] - args.crop_width // 2, 0, maximum_x))
    y = int(np.clip(center[1] - args.crop_height // 2, 0, maximum_y))
    return CropRegion(
        args.source_width,
        args.source_height,
        args.source_fps,
        x,
        y,
        args.crop_width,
        args.crop_height,
    )


def save_region(path: Path, region: CropRegion) -> None:
    validate_region(region)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "camera": "top",
        **asdict(region),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def load_region(path: Path) -> CropRegion | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("camera") != "top":
        raise RuntimeError(f"Invalid top ROI file: {path}")
    try:
        region = CropRegion(
            source_width=int(payload["source_width"]),
            source_height=int(payload["source_height"]),
            source_fps=int(payload["source_fps"]),
            x=int(payload["x"]),
            y=int(payload["y"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
        )
        validate_region(region)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid top ROI file: {path}") from error
    return region


def draw_selection(
    image: np.ndarray,
    region: CropRegion,
    locked: bool,
) -> np.ndarray:
    view = cv2.convertScaleAbs(image, alpha=0.28)
    y0, y1 = region.y, region.y + region.height
    x0, x1 = region.x, region.x + region.width
    view[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    color = (70, 225, 120) if locked else (0, 210, 255)
    cv2.rectangle(view, (x0, y0), (x1 - 1, y1 - 1), color, 3)
    center = (x0 + region.width // 2, y0 + region.height // 2)
    cv2.drawMarker(view, center, color, cv2.MARKER_CROSS, 20, 2)
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (view.shape[1], 64), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, view, 0.28, 0.0, view)
    state = "LOCKED" if locked else "MOVE POINTER"
    cv2.putText(
        view,
        f"TOP ROI {region.width}x{region.height} | {state}",
        (18, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        view,
        "Click center | arrows/WASD nudge | Enter save | R reset | Q cancel",
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (215, 222, 232),
        1,
        cv2.LINE_AA,
    )
    return view


def capture_region(args: argparse.Namespace) -> CropRegion:
    rs: Any = importlib.import_module("pyrealsense2")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.source_width,
        args.source_height,
        rs.format.bgr8,
        15,
    )
    try:
        pipeline.start(config)
    except Exception as error:
        raise RuntimeError(
            "Could not open the top camera. Stop track_obstacle.sh first. "
            f"Camera error: {error}"
        ) from error
    center = [args.source_width // 2, args.source_height // 2]
    selected_center = center.copy()
    locked = False

    def mouse(event: int, x: int, y: int, flags: int, param) -> None:
        nonlocal locked
        del flags, param
        if event == cv2.EVENT_MOUSEMOVE and not locked:
            selected_center[:] = [x, y]
        elif event == cv2.EVENT_LBUTTONDOWN:
            selected_center[:] = [x, y]
            locked = True

    window = "Select fixed top-camera workspace ROI"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.source_width, args.source_height)
    cv2.setMouseCallback(window, mouse)
    try:
        for _ in range(20):
            pipeline.wait_for_frames(3000)
        while True:
            frame = pipeline.wait_for_frames(3000).get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data()).copy()
            region = region_from_center(tuple(selected_center), args)
            cv2.imshow(window, draw_selection(image, region, locked))
            key = cv2.waitKeyEx(20)
            movement = {
                65361: (-5, 0),
                65362: (0, -5),
                65363: (5, 0),
                65364: (0, 5),
                ord("a"): (-5, 0),
                ord("w"): (0, -5),
                ord("d"): (5, 0),
                ord("s"): (0, 5),
            }.get(key)
            if movement is not None:
                selected_center[0] += movement[0]
                selected_center[1] += movement[1]
                locked = True
            elif key in (10, 13):
                return region
            elif key in (ord("r"), ord("R")):
                selected_center[:] = [
                    args.source_width // 2,
                    args.source_height // 2,
                ]
                locked = False
            elif key in (27, ord("q"), ord("Q")):
                raise RuntimeError("Top ROI selection cancelled")
    finally:
        cv2.destroyWindow(window)
        pipeline.stop()


def run_self_test(args: argparse.Namespace) -> None:
    center = region_from_center(
        (args.source_width // 2, args.source_height // 2),
        args,
    )
    left_edge = region_from_center((0, 0), args)
    right_edge = region_from_center((args.source_width, args.source_height), args)
    if (left_edge.x, left_edge.y) != (0, 0):
        raise RuntimeError("ROI did not clamp to the upper-left corner")
    if (
        right_edge.x + right_edge.width != args.source_width
        or right_edge.y + right_edge.height != args.source_height
    ):
        raise RuntimeError("ROI did not clamp to the lower-right corner")
    image = np.full((args.source_height, args.source_width, 3), 180, np.uint8)
    preview = draw_selection(image, center, True)
    if preview.shape != image.shape:
        raise RuntimeError("ROI preview dimensions changed")
    with tempfile.TemporaryDirectory(prefix="top-roi-test-") as directory:
        path = Path(directory) / "top_roi.json"
        save_region(path, center)
        if load_region(path) != center:
            raise RuntimeError("Saved ROI does not reload exactly")
    print(
        f"self-test passed: {center.width}x{center.height} crop at "
        f"{center.x},{center.y}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.self_test:
        run_self_test(args)
        return
    region = capture_region(args)
    save_region(args.output, region)
    print(
        f"Saved top ROI {region.width}x{region.height} at "
        f"{region.x},{region.y} to {args.output}"
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
