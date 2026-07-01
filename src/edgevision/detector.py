"""
edgeVision detector module
--> runs on Jetson Orin Nano A

this file will contain the vision pipeline for camera capture,
object detection, depth estimation, hazard ranking, and TCP alert sending.

"""

#!/usr/bin/env python3

import os

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import argparse
import shutil
import signal
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

from edgevision.camera import Camera
from edgevision.config import (
    CONF_THRESHOLD,
    DEFAULT_THRESHOLD,
    DEPTH_ENGINE,
    DEPTH_MODEL_ID,
    DEPTH_ONNX,
    DEPTH_PERCENTILE,
    DEPTH_SIZE,
    DETECTION_INTERVAL,
    NANO_B_IP,
    NANO_B_PORT,
    SAFETY_THRESHOLDS,
    TRT_WORKSPACE,
    YOLO_IMGSZ,
    YOLO_PT,
    YOLO_TRT,
)
from edgevision.depth import TRTDepth, preprocess_depth
from edgevision.hazard_logic import classify_urgency, direction_of
from edgevision.networking import HazardSender


def log(msg: str) -> None:
    """Print a timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# building the model w/ exports
def _build_yolo_engine() -> None:
    # build the YOLOv8n TensorRT engine from the PyTorch model.
    # this should be run once on Jetson using --build-models.
    from ultralytics import YOLO as _YOLO

    log("Building YOLO TensorRT engine...")
    model = _YOLO(YOLO_PT)
    model.export(format="engine", imgsz=YOLO_IMGSZ, half=True, simplify=True)

    del model
    torch.cuda.empty_cache()

    log("YOLO engine ready")


def _export_depth_onnx() -> None:
    from transformers import AutoModelForDepthEstimation

    log("Exporting Depth Anything V2 to ONNX...")

    class _DepthWrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values).predicted_depth

    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).eval()
    dummy = torch.randn(1, 3, DEPTH_SIZE, DEPTH_SIZE)

    torch.onnx.export(
        _DepthWrapper(model),
        (dummy,),
        DEPTH_ONNX,
        input_names=["pixel_values"],
        output_names=["depth"],
        opset_version=17,
        do_constant_folding=True,
    )

    del model

    log(f"Exported depth model to {DEPTH_ONNX}")


def _build_depth_engine() -> bool:
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"

    if not os.path.exists(trtexec):
        log("trtexec not found - skipping depth TensorRT build.")
        return False

    log("Building Depth Anything TensorRT engine...")

    cmd = [
        trtexec,
        f"--onnx={DEPTH_ONNX}",
        f"--saveEngine={DEPTH_ENGINE}",
        "--fp16",
        f"--memPoolSize=workspace:{TRT_WORKSPACE}",
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0 and os.path.exists(DEPTH_ENGINE):
        log("Depth TensorRT engine ready")
        return True

    log("trtexec failed.")
    return False


def ensure_models() -> None:
    if not os.path.exists(YOLO_TRT):
        _build_yolo_engine()

    if not os.path.exists(DEPTH_ENGINE):
        if not os.path.exists(DEPTH_ONNX):
            _export_depth_onnx()

        _build_depth_engine()


def load_yolo():
    from ultralytics import YOLO

    if not os.path.exists(YOLO_TRT):
        sys.exit(f"YOLO TensorRT engine not found: {YOLO_TRT}. Run --build-models first.")

    log("Loading YOLO TensorRT engine...")
    return YOLO(YOLO_TRT)



# --- DETECTION ---
def detect_hazards( frame_bgr: np.ndarray, yolo,
    depth_runner: TRTDepth,) -> tuple[list[dict], float]:
    
    start_time = time.time()
    height, width = frame_bgr.shape[:2]

    results = yolo(frame_bgr, verbose=False)[0]
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return [], (time.time() - start_time) * 1000.0

    confidences = boxes.conf.cpu().numpy()
    keep_indices = np.where(confidences >= CONF_THRESHOLD)[0]

    if len(keep_indices) == 0:
        return [], (time.time() - start_time) * 1000.0

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    depth_map = depth_runner.infer(preprocess_depth(frame_rgb))

    depth_height, depth_width = depth_map.shape
    scale_x = depth_width / width
    scale_y = depth_height / height

    hazards: list[dict] = []

    for index in keep_indices:
        x1, y1, x2, y2 = map(int, boxes.xyxy[index].tolist())
        label = yolo.names[int(boxes.cls[index])]

        depth_x1 = max(0, int(x1 * scale_x))
        depth_x2 = min(depth_width, int(x2 * scale_x))
        depth_y1 = max(0, int(y1 * scale_y))
        depth_y2 = min(depth_height, int(y2 * scale_y))

        region = depth_map[depth_y1:depth_y2, depth_x1:depth_x2]

        if region.size == 0:
            continue

        distance_m = float(np.percentile(region, DEPTH_PERCENTILE))
        limit = SAFETY_THRESHOLDS.get(label, DEFAULT_THRESHOLD)

        if distance_m <= limit:
            center_x = (x1 + x2) / 2.0
            urgency = classify_urgency(distance_m, limit)

            hazards.append(
                {
                    "label": label,
                    "distance_m": round(distance_m, 1),
                    "direction": direction_of(center_x, width),
                    "urgency": urgency,
                    "conf": float(confidences[index]),
                    "box": (x1, y1, x2, y2),
                }
            )

    hazards.sort(key=lambda hazard: hazard["distance_m"])

    return hazards, (time.time() - start_time) * 1000.0




def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Jetson Nano A: EdgeVision hazard detector"
    )

    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Render debug video overlay. Requires display support.",
    )

    parser.add_argument(
        "--build-models",
        action="store_true",
        help="Build TensorRT engines and exit.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=YOLO_IMGSZ,
        help=f"YOLO input size. Default: {YOLO_IMGSZ}",
    )

    parser.add_argument(
        "--nano-b-ip",
        default=NANO_B_IP,
        help=f"Nano B IP address. Default: {NANO_B_IP}",
    )

    parser.add_argument(
        "--nano-b-port",
        type=int,
        default=NANO_B_PORT,
        help=f"Nano B TCP port. Default: {NANO_B_PORT}",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available. This detector is designed for Jetson GPU inference.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    log(f"Device  : {torch.cuda.get_device_name(0)}")
    log(f"PyTorch : {torch.__version__}")
    log(f"CUDA    : {torch.version.cuda}")

    ensure_models()

    if args.build_models:
        log("--build-models complete. Exiting.")
        return

    yolo = load_yolo()

    if not os.path.exists(DEPTH_ENGINE):
        sys.exit(f"Depth TensorRT engine not found: {DEPTH_ENGINE}. Run --build-models first.")

    depth_runner = TRTDepth(DEPTH_ENGINE)
    log("Depth model: TensorRT ready")

    sender = HazardSender(host=args.nano_b_ip, port=args.nano_b_port)

    log("Warming up models...")

    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    yolo(dummy_bgr, verbose=False)
    depth_runner.infer(preprocess_depth(dummy_rgb))
    torch.cuda.synchronize()

    log("Models ready\n")

    sender.send_ready()

    camera = Camera()

    stop_event = signal_stop_event()

    fps_history = deque(maxlen=30)
    log("Detection running. Ctrl+C to stop.\n")

    try:
        while not stop_event.is_set() and camera.read() is None:
            time.sleep(0.05)

        next_detection = 0.0

        while not stop_event.is_set():
            frame = camera.read()

            if frame is None:
                continue

            now = time.time()

            if now < next_detection:
                time.sleep(0.005)
                continue

            next_detection = now + DETECTION_INTERVAL

            hazards, latency_ms = detect_hazards(frame, yolo, depth_runner)
            fps_history.append(1000.0 / max(latency_ms, 1e-3))

            if hazards:
                top_hazard = hazards[0]
                sender.send(top_hazard)

                log(
                    f"WARN: {top_hazard['label']:<14} "
                    f"{top_hazard['direction']:<13} "
                    f"{top_hazard['distance_m']:>4} m "
                    f"[{top_hazard['urgency']:<8}] "
                    f"[{latency_ms:4.0f} ms | {np.mean(fps_history):4.1f} FPS]"
                )

            if args.show_video:
                show_debug_overlay(frame, hazards)

    finally:
        camera.release()
        sender.close()

        if args.show_video:
            cv2.destroyAllWindows()

        log("Camera released. Bye.")


def signal_stop_event():
    import threading

    stop_event = threading.Event()

    def _on_signal(signum, frame):
        log("\nShutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    return stop_event


def show_debug_overlay(frame: np.ndarray, hazards: list[dict]) -> None:
    visualization = frame.copy()

    for hazard in hazards:
        x1, y1, x2, y2 = hazard["box"]

        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.putText(
            visualization,
            f"{hazard['label']} {hazard['distance_m']}m [{hazard['urgency']}]",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    cv2.imshow("hazards", visualization)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        raise KeyboardInterrupt


if __name__ == "__main__":
    main()