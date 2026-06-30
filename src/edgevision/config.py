"""
configuration settings for edgevision

this file will store any possible model paths, camera settings, network settings,
and hazard detection thresholds used by the detector and receiver modules

"""


# yolo model settings
YOLO_PT = "yolov8n.pt"
YOLO_TRT = "yolov8n.engine"
YOLO_IMGSZ = 640

# depth anything V2 model settings
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_SIZE = 392
DEPTH_ONNX = "depth_anything_v2_metric.onnx"
DEPTH_ENGINE = "depth_anything_v2_metric.engine"
TRT_WORKSPACE = 1024  # MB of scratch space for tensor build

# nano B
NANO_B_IP = "192.168.50.2" # use custom ip if working with your own nanos
NANO_B_PORT = 5005
NANO_B_TIMEOUT = 5.0

# class danger distances in meters
SAFETY_THRESHOLDS = {
    "car": 10.0,
    "truck": 12.0,
    "bus": 12.0,
    "motorcycle": 8.0,
    "bicycle": 6.0,
    "traffic light": 4.0,
    "stop sign": 4.0,
    "fire hydrant": 2.5,
    "parking meter": 2.0,
    "bench": 2.5,
    "person": 3.0,
    "dog": 2.5,
    "cat": 2.0,
    "chair": 2.5,
    "couch": 3.0,
    "bed": 3.0,
    "dining table": 3.0,
    "toilet": 2.0,
    "sink": 2.0,
    "refrigerator": 2.0,
    "tv": 2.0,
    "backpack": 2.0,
    "handbag": 2.0,
    "suitcase": 2.5,
    "potted plant": 2.0,
    "vase": 1.5,
    "bottle": 1.5,
}

# default danger distance for classes not listed above
DEFAULT_THRESHOLD = 3.0

# detection settings
CONF_THRESHOLD = 0.45
DEPTH_PERCENTILE = 25
DETECTION_INTERVAL = 0.5