"""
networking utilities for edgevision

this file will contain helper code for sending hazard alerts
from the vision node to the audio receiver node using TCP messages 

"""

import json
import socket
import threading
import time
from collections import deque

from edgevision.config import NANO_B_TIMEOUT


def log(msg: str) -> None:
    # print a timestamped log message
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HazardSender:
    """
    sends detected hazards from Jetson Nano A to Jetson Nano B over TCP

    runs on a background thread so network communication does not block
    the real-time detection pipeline
    """

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._q: deque[bytes] = deque(maxlen=4)
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._send_loop,
            daemon=True,
            name="hazard-sender",
        )
        self._thread.start()
        log(f"HazardSender --> {host}:{port} (TCP) ready")

    def send(self, hazard: dict) -> None:
       # serialize one hazard as a compact JSON line
        line = (
            json.dumps(
                {
                    "label": hazard["label"],
                    "distance_m": hazard["distance_m"],
                    "direction": hazard["direction"],
                    "urgency": hazard.get("urgency", "low"),
                    "conf": round(hazard["conf"], 3),
                    "ts": time.time(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )

        with self._lock:
            self._q.append(line.encode())

    def send_ready(self) -> None:
        # send msg to announce if ready
        line = (
            json.dumps(
                {
                    "label": "__system__",
                    "message": "Hazard detection system ready.",
                    "ts": time.time(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )

        with self._lock:
            self._q.append(line.encode())

    def close(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        log("HazardSender closed.")

    def _connect(self) -> bool:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(NANO_B_TIMEOUT)
            sock.connect((self.host, self.port))
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = sock
            log(f"HazardSender: connected to Nano B at {self.host}:{self.port}")
            return True
        except OSError as e:
            log(f"HazardSender: could not connect ({e}) - will retry.")
            return False

    def _send_loop(self) -> None:
        while not self._stop.is_set() and not self._connect():
            time.sleep(2.0)

        while not self._stop.is_set():
            payload = None

            with self._lock:
                if self._q:
                    payload = self._q.popleft()

            if payload:
                try:
                    self._sock.sendall(payload)
                except OSError as e:
                    log(f"HazardSender: send error ({e}) - reconnecting...")
                    self._sock = None

                    while not self._stop.is_set() and not self._connect():
                        time.sleep(2.0)
            else:
                time.sleep(0.005)