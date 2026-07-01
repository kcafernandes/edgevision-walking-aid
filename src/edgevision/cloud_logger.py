"""
optional cloud logging utilities for edgevision
--> last priority implementation 

this file will contain privacy-aware logging for detection and
performance records

"""

import json
import os
import queue
import threading
import time


# only these fields are allowed to be logged.
# avoids storing raw images, video frames, or unnecessary private data.
ALLOWED_FIELDS = {
    "label",
    "distance_m",
    "direction",
    "urgency",
    "conf",
    "ts",
    "latency_ms",
}


def _privacy_filter(record: dict) -> dict:
   # only keep the safe fields before logging
    return {key: value for key, value in record.items() if key in ALLOWED_FIELDS}


class CloudLogger:
    
    # optional background logger for detection and performance records.

    def __init__(
        self,
        endpoint: str | None = None,
        local_path: str = "cloud_logs.jsonl",
        batch_size: int = 20,
        flush_interval: float = 5.0,
    ):
        self.endpoint = endpoint
        self.local_path = local_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._q: queue.Queue = queue.Queue()
        self._pending: list[dict] = []
        self._stats = {"total": 0, "sent": 0, "by_label": {}}
        self._stop = threading.Event()

        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="cloud-logger",
        )
        self._thread.start()

    def log_detection(self, hazard: dict, latency_ms: float | None = None) -> None:
        """Log one detected hazard after applying the privacy filter."""
        record = _privacy_filter(dict(hazard))
        record["type"] = "detection"
        record["ts"] = record.get("ts", time.time())

        if latency_ms is not None:
            record["latency_ms"] = round(latency_ms, 1)

        self._q.put(record)

    def log_performance(self, fps: float, latency_ms: float) -> None:
        """Log basic performance information."""
        self._q.put(
            {
                "type": "performance",
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "ts": time.time(),
            }
        )

    def close(self) -> None:
        """Flush remaining logs and stop the background worker."""
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._flush(self._drain())

    def summary(self) -> dict:
        """Return a simple logging summary."""
        return dict(self._stats)

    def _drain(self) -> list[dict]:
        """Pull all waiting records from the queue."""
        items = []

        while not self._q.empty():
            items.append(self._q.get_nowait())

        return items

    def _worker(self) -> None:
        """Batch and flush records in the background."""
        last_flush = time.time()

        while not self._stop.is_set():
            try:
                record = self._q.get(timeout=0.5)
                self._pending.append(record)
                self._stats["total"] += 1

                if record.get("type") == "detection":
                    label = record.get("label", "?")
                    self._stats["by_label"][label] = (
                        self._stats["by_label"].get(label, 0) + 1
                    )

            except queue.Empty:
                pass

            full = len(self._pending) >= self.batch_size
            timed_out = (time.time() - last_flush) >= self.flush_interval

            if self._pending and (full or timed_out):
                self._flush(self._pending)
                self._pending = []
                last_flush = time.time()

    def _flush(self, batch: list[dict]) -> None:
        """Upload or save a batch of records."""
        if not batch:
            return

        try:
            self._upload(batch)
            self._stats["sent"] += len(batch)
        except Exception as error:
            print(f"[cloud] upload failed ({error}), keeping {len(batch)} for retry")
            self._pending = batch + self._pending

    def _upload(self, batch: list[dict]) -> None:
        """Save logs locally or send them to a cloud endpoint."""
        if self.endpoint is None:
            with open(self.local_path, "a") as file:
                for record in batch:
                    file.write(json.dumps(record) + "\n")
        else:
            import requests

            requests.post(self.endpoint, json=batch, timeout=5)


if __name__ == "__main__":
    cloud = CloudLogger(local_path="cloud_logs.jsonl", batch_size=3, flush_interval=1.0)

    cloud.log_detection(
        {
            "label": "car",
            "distance_m": 4.0,
            "direction": "on your right",
            "urgency": "high",
            "conf": 0.91,
            "box": (1, 2, 3, 4),
        },
        latency_ms=120,
    )

    cloud.log_detection(
        {
            "label": "person",
            "distance_m": 2.0,
            "direction": "ahead",
            "urgency": "critical",
            "conf": 0.88,
        },
        latency_ms=118,
    )

    cloud.log_performance(fps = 8.3, latency_ms = 120)
    time.sleep(1.5)
    cloud.close()

    print("stats:", cloud.summary())
    print("wrote log file:", os.path.abspath("cloud_logs.jsonl"))