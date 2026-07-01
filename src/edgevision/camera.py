"""
camera utilities for edgevision 

this file will contain CSI camera capture code using GStreamer
for reading real-time frames on the jetson vision node

"""

import threading

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst


Gst.init(None)


class Camera:   
    """
    csi camera wrapper

    keeps the newest frame only so the detection pipeline does not fall behind real time
    
    """

    def __init__(
        self,
        sensor_id: int = 0,
        capture_w: int = 1280,
        capture_h: int = 720,
        display_w: int = 640,
        display_h: int = 480,
        framerate: int = 30,
        flip_method: int = 0,
    ):
        self.frame = None
        self._lock = threading.Lock()
        self._w = display_w
        self._h = display_h

        pipe = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM),width={capture_w},height={capture_h},"
            f"framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw,width={display_w},height={display_h},format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink name=sink max-buffers=1 drop=true sync=false emit-signals=true"
        )

        self._pipeline = Gst.parse_launch(pipe)
        sink = self._pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_frame)

        ret = self._pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("CSI pipeline failed")

        print("CSI camera started", flush=True)

    #pull the newest camera frame from the GStreamer buffer
    def _on_frame(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)

        if ok:
            frame = (
                np.frombuffer(info.data, dtype=np.uint8)
                .reshape(self._h, self._w, 3)
                .copy()
            )

            buf.unmap(info)

            with self._lock:
                self.frame = frame

        return Gst.FlowReturn.OK

    # return the latest camera freame
    def read(self) -> np.ndarray | None:
        with self._lock:
            return self.frame

    # release the cam pipeline
    def release(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)