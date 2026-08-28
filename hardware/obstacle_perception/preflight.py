from __future__ import annotations

import argparse
import os
import platform
import sys

MODEL_ID = "facebook/sam2.1-hiera-small"
TOP_CAMERA_SERIAL = "323622271046"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--skip-display", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--serial", default=TOP_CAMERA_SERIAL)
    return parser.parse_args()


def check_imports() -> tuple[object, object]:
    import cv2
    import numpy
    import pyrealsense2
    import rerun
    import torch
    import transformers

    print(f"python {platform.python_version()}")
    print(f"numpy {numpy.__version__}")
    print(f"opencv {cv2.__version__}")
    print(f"torch {torch.__version__}")
    print(f"transformers {transformers.__version__}")
    print(f"rerun {rerun.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    else:
        print("SAM2 will run on CPU; this is acceptable for the static prototype")
    return pyrealsense2, torch


def check_display() -> None:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "No graphical display found. Run from the desktop or use SSH X forwarding"
        )
    print("graphical display available")


def check_camera(rs, serial: str) -> None:
    devices = list(rs.context().query_devices())
    found = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        device_serial = device.get_info(rs.camera_info.serial_number)
        found.append(device_serial)
        print(f"camera: {name} [{device_serial}]")
    if serial not in found:
        raise RuntimeError(
            f"Top RealSense {serial} not found. Connected serials: {found or 'none'}"
        )
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    try:
        frames = None
        for _ in range(10):
            frames = pipeline.wait_for_frames(5000)
        if (
            frames is None
            or not frames.get_depth_frame()
            or not frames.get_color_frame()
        ):
            raise RuntimeError("RealSense did not return an RGB-D frameset")
    finally:
        pipeline.stop()
    print(f"top camera {serial} returned RGB-D frames")


def download_model(model_id: str) -> None:
    from transformers import Sam2Model, Sam2Processor

    print(f"caching {model_id}")
    Sam2Processor.from_pretrained(model_id)
    Sam2Model.from_pretrained(model_id)
    print(f"cached {model_id}")


def main() -> None:
    args = parse_args()
    rs, _ = check_imports()
    if not args.skip_display:
        check_display()
    if args.download_model:
        download_model(args.model)
    if not args.skip_camera:
        check_camera(rs, args.serial)
    print("preflight passed")


if __name__ == "__main__":
    main()
