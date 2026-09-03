from __future__ import annotations

import argparse
import importlib
import io
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from camera_web import LiveState, start_dashboard
from point_cloud_stream import (
    PointCloudStream,
    draw_gripper_marker,
    draw_projected_camera_box,
    draw_projected_policy_trajectories,
)
from policy_trajectory_stream import DemoTrajectoryStream
from select_top_roi import CropRegion, load_region

TOP_CAMERA_SERIAL = "323622271046"
LEFT_CAMERA_SERIAL = "335122272969"
MODEL_ID = "facebook/sam2.1-hiera-small"
SCRIPT_DIR = Path(__file__).resolve().parent
TARGET_DIR = SCRIPT_DIR / "calibration" / "targets"
CAMERA_CALIBRATION_PATH = SCRIPT_DIR / "calibration" / "top_from_left.json"
ROBOT_CALIBRATION_PATH = SCRIPT_DIR / "calibration" / "base_from_top.json"
TOP_ROI_PATH = SCRIPT_DIR / "calibration" / "top_roi.json"
LEGACY_TARGET_DIR = SCRIPT_DIR / "runs" / "tracking_targets"
LEGACY_PUBLISH_DIR = Path("/dev/shm/flow_obstacle_tracking")


def parse_point(raw: str) -> tuple[float, float]:
    try:
        x, y = raw.split(",", 1)
        return float(x), float(y)
    except ValueError as error:
        raise argparse.ArgumentTypeError("point must be X,Y") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-serial", default=TOP_CAMERA_SERIAL)
    parser.add_argument("--left-serial", default=LEFT_CAMERA_SERIAL)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--tracking-fps", type=float, default=5.0)
    parser.add_argument("--raw-fps", type=float, default=10.0)
    parser.add_argument("--match-threshold", type=float, default=0.82)
    parser.add_argument("--match-min-scale", type=float, default=0.4)
    parser.add_argument("--match-max-scale", type=float, default=2.4)
    parser.add_argument("--session-frames", type=int, default=120)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-point", type=parse_point)
    parser.add_argument("--left-point", type=parse_point)
    parser.add_argument("--target-dir", type=Path, default=TARGET_DIR)
    parser.add_argument("--top-roi", type=Path, default=TOP_ROI_PATH)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=CAMERA_CALIBRATION_PATH,
    )
    parser.add_argument(
        "--robot-calibration",
        type=Path,
        default=ROBOT_CALIBRATION_PATH,
    )
    parser.add_argument("--can-interface", default="can_arm_right")
    parser.add_argument("--demo-trajectory-count", type=int, default=3)
    parser.add_argument("--select-targets", action="store_true")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def intrinsics_to_dict(intrinsics) -> dict[str, object]:
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


def crop_rgbd(
    image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, object],
    region: CropRegion,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if image.shape[:2] != (region.source_height, region.source_width):
        raise RuntimeError(
            f"Top color frame is {image.shape[1]}x{image.shape[0]}, expected "
            f"{region.source_width}x{region.source_height}"
        )
    if depth_m.shape != image.shape[:2]:
        raise RuntimeError("Aligned top color and depth dimensions differ")
    y0, y1 = region.y, region.y + region.height
    x0, x1 = region.x, region.x + region.width
    cropped_intrinsics = dict(intrinsics)
    cropped_intrinsics.update(
        {
            "width": region.width,
            "height": region.height,
            "ppx": cast(float, intrinsics["ppx"]) - region.x,
            "ppy": cast(float, intrinsics["ppy"]) - region.y,
        }
    )
    return (
        image[y0:y1, x0:x1].copy(),
        depth_m[y0:y1, x0:x1].copy(),
        cropped_intrinsics,
    )


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    with temporary.open("wb") as output:
        output.write(payload)
    os.replace(temporary, path)


def encode_image(extension: str, image: np.ndarray, quality: int = 85) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension == ".jpg" else []
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode {extension} image")
    return encoded.tobytes()


class StatusBoard:
    def __init__(self, live_state: LiveState):
        self.live_state = live_state
        self.run_id = uuid.uuid4().hex[:12]
        self.phase = "Starting"
        self.cameras = {
            name: {
                "tracking": False,
                "message": "Starting camera",
                "fps": 0.0,
                "mask_area": 0,
            }
            for name in ("top", "left")
        }
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.publish_loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.publish(False)

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase

    def set_camera(self, name: str, **values) -> None:
        with self.lock:
            self.cameras[name].update(values)

    def publish(self, running: bool = True) -> None:
        with self.lock:
            payload = {
                "running": running,
                "run_id": self.run_id,
                "updated_at": time.time(),
                "phase": self.phase,
                "cameras": self.cameras,
                "outlier_trim_percent": self.live_state.outlier_trim_percent(),
                "policy": self.live_state.policy_status(),
                "policy_trajectories_visible": (
                    self.live_state.policy_trajectory_visibility()
                ),
            }
        self.live_state.publish_status(payload)

    def publish_loop(self) -> None:
        while not self.stop_event.wait(0.25):
            self.publish()


class CameraCapture:
    def __init__(
        self,
        name: str,
        serial: str,
        args: argparse.Namespace,
        board: StatusBoard,
        live_state: LiveState,
        crop: CropRegion | None = None,
    ):
        self.name = name
        self.serial = serial
        self.args = args
        self.board = board
        self.live_state = live_state
        self.crop = crop
        self.image: np.ndarray | None = None
        self.depth_m: np.ndarray | None = None
        self.intrinsics: dict[str, object] | None = None
        self.source_image: np.ndarray | None = None
        self.source_intrinsics: dict[str, object] | None = None
        self.version = 0
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.capture, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        self.thread.join(timeout=5.0)

    def latest(
        self,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        dict[str, object] | None,
        np.ndarray | None,
        dict[str, object] | None,
        int,
    ]:
        with self.condition:
            image = None if self.image is None else self.image.copy()
            depth_m = None if self.depth_m is None else self.depth_m.copy()
            intrinsics = None if self.intrinsics is None else dict(self.intrinsics)
            source_image = (
                None if self.source_image is None else self.source_image.copy()
            )
            source_intrinsics = (
                None if self.source_intrinsics is None else dict(self.source_intrinsics)
            )
            return (
                image,
                depth_m,
                intrinsics,
                source_image,
                source_intrinsics,
                self.version,
            )

    def capture(self) -> None:
        rs: Any = importlib.import_module("pyrealsense2")
        raw_period = 1.0 / self.args.raw_fps
        last_raw = 0.0
        width = self.crop.source_width if self.crop is not None else self.args.width
        height = self.crop.source_height if self.crop is not None else self.args.height
        fps = self.crop.source_fps if self.crop is not None else self.args.fps
        while not self.stop_event.is_set():
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.depth,
                width,
                height,
                rs.format.z16,
                fps,
            )
            config.enable_stream(
                rs.stream.color,
                width,
                height,
                rs.format.bgr8,
                fps,
            )
            started = False
            try:
                profile = pipeline.start(config)
                started = True
                align = rs.align(rs.stream.color)
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
                camera_message = f"Camera live · {width}x{height}@{fps}"
                if self.crop is not None:
                    camera_message += (
                        f" → {self.crop.width}x{self.crop.height} ROI at "
                        f"{self.crop.x},{self.crop.y}"
                    )
                self.board.set_camera(self.name, message=camera_message)
                while not self.stop_event.is_set():
                    frames = align.process(pipeline.wait_for_frames(3000))
                    color = frames.get_color_frame()
                    depth = frames.get_depth_frame()
                    if not color or not depth:
                        continue
                    image = np.asanyarray(color.get_data()).copy()
                    depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * scale
                    intrinsics = intrinsics_to_dict(
                        depth.profile.as_video_stream_profile().intrinsics
                    )
                    source_image = image
                    source_intrinsics = intrinsics
                    if self.crop is not None:
                        image, depth_m, intrinsics = crop_rgbd(
                            image,
                            depth_m,
                            intrinsics,
                            self.crop,
                        )
                    with self.condition:
                        self.image = image
                        self.depth_m = depth_m
                        self.intrinsics = intrinsics
                        self.source_image = source_image
                        self.source_intrinsics = source_intrinsics
                        self.version += 1
                        self.condition.notify_all()
                    now = time.monotonic()
                    if now - last_raw >= raw_period:
                        self.live_state.publish_image(
                            f"{self.name}_raw.jpg",
                            encode_image(
                                ".jpg",
                                source_image,
                                self.args.jpeg_quality,
                            ),
                        )
                        last_raw = now
            except Exception as error:
                self.board.set_camera(
                    self.name,
                    tracking=False,
                    message=f"Camera error: {error}",
                )
            finally:
                if started:
                    pipeline.stop()
            self.stop_event.wait(1.0)


def prompt_views(
    cameras: dict[str, CameraCapture],
    supplied: dict[str, tuple[float, float] | None],
) -> dict[str, tuple[float, float]]:
    points = {name: supplied[name] for name in ("top", "left")}
    if all(point is not None for point in points.values()):
        return {name: point for name, point in points.items() if point is not None}
    window = "Select cups: TOP first, then LEFT | r=reset q=cancel"
    width = cameras["top"].args.width

    def mouse(event: int, x: int, y: int, flags: int, param) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if points["top"] is None and x < width:
            points["top"] = (float(x), float(y))
        elif points["left"] is None and x >= width:
            points["left"] = (float(x - width), float(y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 2 * width, cameras["top"].args.height)
    cv2.setMouseCallback(window, mouse)
    print("Click the cups in the TOP image, then click them in the LEFT image")
    while not all(point is not None for point in points.values()):
        images = []
        ready = True
        for name in ("top", "left"):
            image = cameras[name].latest()[0]
            if image is None:
                ready = False
                break
            view = image.copy()
            active = points[name] is None and all(
                points[previous] is not None
                for previous in (("top",) if name == "left" else ())
            )
            colour = (0, 220, 255) if active else (120, 120, 120)
            cv2.rectangle(
                view, (1, 1), (view.shape[1] - 2, view.shape[0] - 2), colour, 3
            )
            cv2.putText(
                view,
                name.upper(),
                (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                colour,
                2,
                cv2.LINE_AA,
            )
            if points[name] is not None:
                point = points[name]
                assert point is not None
                center = (round(point[0]), round(point[1]))
                cv2.circle(view, center, 6, (0, 255, 0), -1)
            images.append(view)
        if ready:
            cv2.imshow(window, np.hstack(images))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points = {"top": supplied["top"], "left": supplied["left"]}
        elif key in (27, ord("q")):
            cv2.destroyWindow(window)
            raise RuntimeError("Target selection cancelled")
    cv2.waitKey(250)
    cv2.destroyWindow(window)
    return {name: point for name, point in points.items() if point is not None}


def target_reference(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return np.zeros((1, 1, 4), np.uint8)
    pad = 20
    y0, y1 = max(int(rows.min()) - pad, 0), min(int(rows.max()) + pad + 1, len(mask))
    x0, x1 = (
        max(int(cols.min()) - pad, 0),
        min(int(cols.max()) + pad + 1, mask.shape[1]),
    )
    crop = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2BGRA)
    crop[..., 3] = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    return crop


def locate_target(
    image: np.ndarray,
    reference: np.ndarray,
    min_scale: float,
    max_scale: float,
) -> tuple[np.ndarray, float, tuple[int, int, int, int]]:
    if reference.shape[2] != 4:
        raise RuntimeError("Saved target reference must have an alpha channel")
    best_score = float("-inf")
    best_location = (0, 0)
    best_alpha: np.ndarray | None = None
    for scale in np.linspace(min_scale, max_scale, 25):
        width = round(reference.shape[1] * scale)
        height = round(reference.shape[0] * scale)
        if width < 8 or height < 8 or width > image.shape[1] or height > image.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(reference, (width, height), interpolation=interpolation)
        template = resized[..., :3]
        alpha = resized[..., 3]
        if np.count_nonzero(alpha) < 16:
            continue
        scores = cv2.matchTemplate(
            image,
            template,
            cv2.TM_CCORR_NORMED,
            mask=alpha,
        )
        scores = np.where(np.isfinite(scores), scores, -np.inf).astype(np.float32)
        _, score, _, location = cv2.minMaxLoc(scores)
        if score > best_score:
            best_score = score
            best_location = location
            best_alpha = alpha
    if best_alpha is None:
        raise RuntimeError("Saved target is larger than the current camera frame")
    x, y = best_location
    height, width = best_alpha.shape
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[y : y + height, x : x + width] = best_alpha >= 128
    return mask, best_score, (x, y, width, height)


def encode_target(
    prompt: tuple[float, float], image: np.ndarray, mask: np.ndarray
) -> bytes:
    output = io.BytesIO()
    np.savez_compressed(
        output,
        mask=mask,
        prompt_xy=np.asarray(prompt, np.float32),
        reference=target_reference(image, mask),
    )
    return output.getvalue()


def save_target(
    target_dir: Path,
    name: str,
    prompt: tuple[float, float],
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    reference = target_reference(image, mask)
    atomic_write(target_dir / f"{name}.npz", encode_target(prompt, image, mask))
    atomic_write(
        target_dir / f"{name}_reference.png",
        encode_image(".png", reference),
    )


def persist_loaded_target(
    target_dir: Path, name: str, target: dict[str, object]
) -> None:
    mask = target["mask"]
    prompt = cast(tuple[float, float], target["prompt"])
    reference = target["reference"]
    assert isinstance(mask, np.ndarray)
    assert isinstance(reference, np.ndarray)
    output = io.BytesIO()
    np.savez_compressed(
        output,
        mask=mask,
        prompt_xy=np.asarray(prompt, np.float32),
        reference=reference,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(target_dir / f"{name}.npz", output.getvalue())
    atomic_write(
        target_dir / f"{name}_reference.png",
        encode_image(".png", reference),
    )


def load_target(path: Path) -> dict[str, object]:
    with np.load(path) as target:
        mask = target["mask"].astype(bool)
        prompt_values = target["prompt_xy"].astype(float).tolist()
        reference = target["reference"].copy()
    if mask.ndim != 2 or len(prompt_values) != 2 or reference.ndim != 3:
        raise RuntimeError(f"Invalid saved target: {path}")
    return {
        "mask": mask,
        "prompt": (float(prompt_values[0]), float(prompt_values[1])),
        "reference": reference,
    }


def migrate_published_target(target_dir: Path, name: str) -> bool:
    source = LEGACY_PUBLISH_DIR / f"{name}_latest.npz"
    if not source.exists():
        return False
    with np.load(source) as latest:
        mask = latest["mask"].astype(bool)
        prompt_values = latest["prompt_xy"].astype(float).tolist()
        rgb = latest["rgb"].copy()
    if not mask.any() or len(prompt_values) != 2:
        return False
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    prompt = (float(prompt_values[0]), float(prompt_values[1]))
    save_target(target_dir, name, prompt, image, mask)
    return True


def load_targets(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    targets = {}
    for name in ("top", "left"):
        path = args.target_dir / f"{name}.npz"
        if not path.exists():
            legacy = LEGACY_TARGET_DIR / f"{name}.npz"
            if legacy.exists():
                persist_loaded_target(args.target_dir, name, load_target(legacy))
            else:
                migrate_published_target(args.target_dir, name)
        if not path.exists():
            raise RuntimeError(
                "No saved obstacle targets. Run select_obstacle.sh once first."
            )
        targets[name] = load_target(path)
    return targets


def load_sam(args: argparse.Namespace):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = Sam2VideoProcessor.from_pretrained(args.model)
    model: Any = Sam2VideoModel.from_pretrained(args.model)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, processor, device, dtype


class ViewTracker:
    def __init__(
        self,
        name: str,
        prompt: tuple[float, float],
        args: argparse.Namespace,
        live_state: LiveState,
        crop: CropRegion | None = None,
    ):
        self.name = name
        self.prompt = prompt
        self.args = args
        self.live_state = live_state
        self.crop = crop
        self.session = None
        self.bootstrap_mask: np.ndarray | None = None
        self.last_output_time = 0.0
        self.output_fps = 0.0

    def new_session(self, processor, device, dtype, shape):
        session = processor.init_video_session(
            inference_device=device,
            inference_state_device="cpu",
            processing_device="cpu",
            video_storage_device="cpu",
            max_vision_features_cache_size=1,
            dtype=dtype,
        )
        height, width = shape
        if self.bootstrap_mask is None:
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=0,
                obj_ids=1,
                input_points=[[[[self.prompt[0], self.prompt[1]]]]],
                input_labels=[[[1]]],
                original_size=(height, width),
            )
        else:
            session.video_height = height
            session.video_width = width
            processor.add_inputs_to_inference_session(
                session,
                frame_idx=0,
                obj_ids=1,
                input_masks=self.bootstrap_mask,
            )
        return session

    def publish(
        self,
        image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: dict[str, object],
        mask: np.ndarray,
        source_image: np.ndarray | None = None,
    ) -> None:
        overlay = image.copy() if source_image is None else source_image.copy()
        display_mask = mask
        marker = self.live_state.gripper_marker(self.name)
        camera_box = self.live_state.camera_box(self.name)
        camera_trajectories = self.live_state.camera_trajectories(self.name)
        pickup_steps = self.live_state.policy_pickup_steps()
        if self.crop is not None and source_image is not None:
            display_mask = np.zeros(overlay.shape[:2], dtype=bool)
            y0, y1 = self.crop.y, self.crop.y + self.crop.height
            x0, x1 = self.crop.x, self.crop.x + self.crop.width
            display_mask[y0:y1, x0:x1] = mask
            if marker is not None:
                marker = (
                    marker[0] + self.crop.x,
                    marker[1] + self.crop.y,
                    marker[2],
                )
            if camera_box is not None:
                camera_box = tuple(
                    (point[0] + self.crop.x, point[1] + self.crop.y)
                    for point in camera_box
                )
            camera_trajectories = tuple(
                tuple(
                    (point[0] + self.crop.x, point[1] + self.crop.y)
                    for point in trajectory
                )
                for trajectory in camera_trajectories
            )
        green = np.zeros_like(overlay)
        green[..., 1] = 255
        overlay[display_mask] = cv2.addWeighted(
            overlay[display_mask],
            0.5,
            green[display_mask],
            0.5,
            0.0,
        )
        if self.crop is not None and source_image is not None:
            roi_overlay = overlay.copy()
            cv2.rectangle(
                roi_overlay,
                (self.crop.x, self.crop.y),
                (
                    self.crop.x + self.crop.width - 1,
                    self.crop.y + self.crop.height - 1,
                ),
                (170, 215, 235),
                2,
                cv2.LINE_AA,
            )
            cv2.addWeighted(roi_overlay, 0.35, overlay, 0.65, 0.0, overlay)
        draw_projected_policy_trajectories(
            overlay,
            camera_trajectories,
            float(overlay.shape[1]),
            float(overlay.shape[0]),
            pickup_steps,
        )
        draw_projected_camera_box(
            overlay,
            camera_box,
            float(overlay.shape[1]),
            float(overlay.shape[0]),
        )
        draw_gripper_marker(
            overlay,
            marker,
            float(overlay.shape[1]),
            float(overlay.shape[0]),
        )
        self.live_state.publish_image(
            f"{self.name}_overlay.jpg",
            encode_image(".jpg", overlay, self.args.jpeg_quality),
        )
        self.live_state.publish_image(
            f"{self.name}_mask.png",
            encode_image(".png", mask.astype(np.uint8) * 255),
        )
        self.live_state.publish_sample(
            self.name,
            {
                "rgb": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                "depth_m": depth_m,
                "mask": mask,
                "intrinsics": intrinsics,
            },
        )

    def process(
        self,
        model,
        processor,
        device,
        dtype,
        image: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: dict[str, object],
        source_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        import torch

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        if self.session is None:
            self.session = self.new_session(processor, device, dtype, (height, width))
        pixel_values = processor.image_processor(images=rgb, return_tensors="pt")[
            "pixel_values"
        ][0]
        with torch.inference_mode():
            output = model(self.session, frame=pixel_values)
        masks = processor.post_process_masks(
            output.pred_masks.float().cpu().unsqueeze(0),
            [(height, width)],
        )
        mask = masks[0][0, 0].numpy()
        score = float(output.object_score_logits.float().cpu().flatten()[0])
        visible_mask = mask if score > 0.0 else np.zeros_like(mask)
        self.publish(
            image,
            depth_m,
            intrinsics,
            visible_mask,
            source_image,
        )
        now = time.monotonic()
        if self.last_output_time:
            instant = 1.0 / max(now - self.last_output_time, 1e-6)
            self.output_fps = 0.8 * self.output_fps + 0.2 * instant
        self.last_output_time = now
        if self.session.num_frames >= self.args.session_frames:
            self.bootstrap_mask = mask
            self.session = None
        return visible_mask, score, self.output_fps


def select_targets(
    args: argparse.Namespace,
    cameras: dict[str, CameraCapture],
    board: StatusBoard,
) -> None:
    board.set_phase("Select the cups in both camera views")
    prompts = prompt_views(
        cameras,
        {"top": args.top_point, "left": args.left_point},
    )
    for name in cameras:
        board.set_camera(name, message="Target selected; creating cutout")
    board.set_phase("Creating saved target cutouts")
    model, processor, device, dtype = load_sam(args)
    for name, camera in cameras.items():
        image, depth_m, intrinsics, _, _, _ = camera.latest()
        if image is None or depth_m is None or intrinsics is None:
            raise RuntimeError(f"{name} camera did not provide an RGB-D frame")
        tracker = ViewTracker(
            name,
            prompts[name],
            args,
            board.live_state,
            camera.crop,
        )
        mask, score, _ = tracker.process(
            model,
            processor,
            device,
            dtype,
            image,
            depth_m,
            intrinsics,
        )
        if score <= 0.0 or not mask.any():
            raise RuntimeError(f"SAM2 did not create a valid {name} target")
        save_target(args.target_dir, name, prompts[name], image, mask)
        board.set_camera(
            name,
            tracking=True,
            message="Target cutout saved",
            mask_area=int(mask.sum()),
        )
    board.set_phase("Targets saved")
    board.publish()
    print(f"Saved top and left targets in {args.target_dir}")
    print("Run track_obstacle.sh to start tracking")


def align_trackers(
    args: argparse.Namespace,
    cameras: dict[str, CameraCapture],
    board: StatusBoard,
    targets: dict[str, dict[str, object]],
) -> dict[str, ViewTracker]:
    trackers = {}
    for name in cameras:
        prompt = cast(tuple[float, float], targets[name]["prompt"])
        reference = targets[name]["reference"]
        assert isinstance(reference, np.ndarray)
        deadline = time.monotonic() + 10.0
        image = cameras[name].latest()[0]
        while image is None and time.monotonic() < deadline:
            time.sleep(0.05)
            image = cameras[name].latest()[0]
        if image is None:
            raise RuntimeError(f"{name} camera did not provide a color frame")
        mask, score, box = locate_target(
            image,
            reference,
            args.match_min_scale,
            args.match_max_scale,
        )
        if score < args.match_threshold:
            raise RuntimeError(
                f"Could not find {name} target cutout: match {score:.3f} is below "
                f"{args.match_threshold:.3f}"
            )
        tracker = ViewTracker(
            name,
            prompt,
            args,
            board.live_state,
            cameras[name].crop,
        )
        tracker.bootstrap_mask = mask
        trackers[name] = tracker
        x, y, width, height = box
        board.set_camera(
            name,
            message=(f"Target found at {x},{y} ({width}x{height}, match {score:.2f})"),
            match_score=round(score, 3),
            aligned_at=time.time(),
        )
    return trackers


def track_targets(
    args: argparse.Namespace,
    cameras: dict[str, CameraCapture],
    board: StatusBoard,
    targets: dict[str, dict[str, object]],
) -> None:
    for name in cameras:
        board.set_camera(name, message="Saved target loaded; loading SAM2")
    board.set_phase("Loading SAM2 with saved targets")
    model, processor, device, dtype = load_sam(args)
    board.set_phase("Aligning saved targets")
    trackers = align_trackers(args, cameras, board, targets)
    versions = {name: -1 for name in cameras}
    period = 1.0 / args.tracking_fps
    board.set_phase("Tracking both camera views")
    while True:
        cycle_start = time.monotonic()
        if board.live_state.consume_realign_request():
            board.set_phase("Reloading and realigning saved targets")
            try:
                reloaded_targets = load_targets(args)
                aligned_trackers = align_trackers(
                    args,
                    cameras,
                    board,
                    reloaded_targets,
                )
                trackers = aligned_trackers
                versions = {name: -1 for name in cameras}
                board.set_phase("Tracking both camera views")
            except Exception as error:
                board.set_phase(f"Realignment failed: {error}")
        for name, camera in cameras.items():
            (
                image,
                depth_m,
                intrinsics,
                source_image,
                _,
                version,
            ) = camera.latest()
            if (
                image is None
                or depth_m is None
                or intrinsics is None
                or version == versions[name]
            ):
                continue
            versions[name] = version
            try:
                mask, score, output_fps = trackers[name].process(
                    model,
                    processor,
                    device,
                    dtype,
                    image,
                    depth_m,
                    intrinsics,
                    source_image,
                )
                board.set_camera(
                    name,
                    tracking=score > 0.0,
                    message=(
                        "Tracking target"
                        if score > 0.0
                        else "Target temporarily not visible"
                    ),
                    fps=round(output_fps, 1),
                    mask_area=int(mask.sum()),
                )
            except Exception as error:
                trackers[name].session = None
                trackers[name].bootstrap_mask = None
                board.set_camera(
                    name,
                    tracking=False,
                    message=f"Tracking error: {error}",
                    fps=0.0,
                    mask_area=0,
                )
        delay = period - (time.monotonic() - cycle_start)
        if delay > 0:
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    if args.tracking_fps <= 0 or args.raw_fps <= 0:
        raise ValueError("FPS values must be positive")
    if not 0.0 <= args.match_threshold <= 1.0:
        raise ValueError("Match threshold must be between zero and one")
    if not 0.0 < args.match_min_scale <= args.match_max_scale:
        raise ValueError("Match scale range must be positive and ordered")
    top_roi = load_region(args.top_roi)
    if top_roi is not None and (top_roi.width, top_roi.height) != (
        args.width,
        args.height,
    ):
        raise ValueError(
            f"Top ROI must output {args.width}x{args.height}; got "
            f"{top_roi.width}x{top_roi.height}"
        )
    targets = None if args.select_targets else load_targets(args)
    live_state = LiveState()
    dashboard = None
    cloud_stream = None
    policy_stream = None
    if not args.select_targets:
        cloud_stream = PointCloudStream(
            live_state,
            args.jpeg_quality,
            args.camera_calibration,
            args.robot_calibration,
            args.top_serial,
            args.left_serial,
            args.can_interface,
        )
        cloud_stream.start()
        policy_stream = DemoTrajectoryStream(
            live_state,
            args.demo_trajectory_count,
        )
        policy_stream.start()
        dashboard = start_dashboard(
            live_state,
            args.target_dir,
            args.bind,
            args.port,
        )
        print(f"Open http://{socket.gethostname()}:{args.port}")
    board = StatusBoard(live_state)
    cameras = {
        "top": CameraCapture(
            "top",
            args.top_serial,
            args,
            board,
            live_state,
            top_roi,
        ),
        "left": CameraCapture("left", args.left_serial, args, board, live_state),
    }
    board.start()
    for camera in cameras.values():
        camera.start()
    try:
        if args.select_targets:
            select_targets(args, cameras, board)
        else:
            assert targets is not None
            track_targets(args, cameras, board, targets)
    except KeyboardInterrupt:
        board.set_phase("Stopped")
    finally:
        for camera in cameras.values():
            camera.stop()
        cv2.destroyAllWindows()
        board.stop()
        if cloud_stream is not None:
            cloud_stream.stop()
        if policy_stream is not None:
            policy_stream.stop()
        if dashboard is not None:
            dashboard.shutdown()
            dashboard.server_close()


if __name__ == "__main__":
    main()
