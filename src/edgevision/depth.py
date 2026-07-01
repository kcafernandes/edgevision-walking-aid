"""
depth estimation utilities for edgevision

this file will contain preprocessing and TensorRT inference code
for running Depth Anything V2 metric depth estimation on jetson

"""

import numpy as np
import cv2
import torch

from edgevision.config import DEPTH_SIZE


# numpy dtype -> torch dtype lookup, used when allocating TensorRT I/O buffers
_NP2TORCH = {
    np.float32: torch.float32,
    np.float16: torch.float16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.bool_: torch.bool,
}

# imageNet normalization consts used by Depth Anything V2
_DEPTH_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_DEPTH_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

# converts the image to the expected size, scales pixel values,
# changes layout from HWC to CHW, and applies the prev normalization
def preprocess_depth(frame_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(
        frame_rgb,
        (DEPTH_SIZE, DEPTH_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = (img - _DEPTH_MEAN) / _DEPTH_STD

    return np.ascontiguousarray(img[None])


#wrapper to for running depth on jetson
class TRTDepth:

    def __init__(self, engine_path: str, device: str = "cuda"):
        import tensorrt as trt

        self.device = device
        logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.buffers: dict[str, torch.Tensor] = {}
        self.in_name = ""
        self.out_name = ""
        self.in_dtype = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = _NP2TORCH[np_dtype]

            buffer = torch.empty(shape, dtype=torch_dtype, device=device)
            self.buffers[name] = buffer
            self.context.set_tensor_address(name, buffer.data_ptr())

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
                self.in_dtype = torch_dtype
            else:
                self.out_name = name

        self.stream = torch.cuda.Stream()

        print(
            f"Depth TRT | in {tuple(self.buffers[self.in_name].shape)}"
            f" --> out {tuple(self.buffers[self.out_name].shape)}",
            flush=True,
        )

    @torch.no_grad() 
    # return a depth map in meters
    def infer(self, chw_f32: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(chw_f32).to(
            self.device,
            self.in_dtype,
            non_blocking=True,
        )

        self.buffers[self.in_name].copy_(x)
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        return self.buffers[self.out_name].squeeze().float().cpu().numpy()