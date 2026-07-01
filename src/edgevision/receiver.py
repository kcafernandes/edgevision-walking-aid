"""
edgeVision receiver module
--> runs on Jetson Orin Nano B

this file will receive hazard alerts from the detector node,
prioritize urgent warnings, and convert them into audio/TTS feedback.

"""

#!/usr/bin/env python3

import argparse
import json
import queue
import socket
import subprocess
import threading
import time


# network settings
LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 5005

# base wait time before repeating the same alert type
ALERT_COOLDOWN = 2.5

# critical alerts can repeat sooner while low-priority alerts wait longer
URGENCY_COOLDOWN = {
    "critical": ALERT_COOLDOWN * 0.6,
    "high": ALERT_COOLDOWN,
    "low": ALERT_COOLDOWN * 2.0,
}

URGENCY_RANK = {
    "critical": 0,
    "high": 1,
    "low": 2,
}

# optional based on device
ALSA_DEVICE = None

# priority queue for incoming alerts
_alert_queue: queue.PriorityQueue = queue.PriorityQueue()

# tracks when each alert label was last spoken
_last_spoken: dict[str, float] = {}

# prints the timestamp msgs
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# organized speak functions that sends alerts in an organized matter so they don't overlap
def speak_blocking(text: str) -> None:
    cmd = ["espeak-ng", "-s", "145", "-a", "200"]

    if ALSA_DEVICE:
        cmd += ["-d", ALSA_DEVICE]

    cmd.append(text)

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        log(f"TTS is unavailable but would have said: {text}")


# converts the hazard packets into an understandable phrase
def build_phrase(alert: dict) -> str:
    label = alert["label"]
    distance_m = alert.get("distance_m", 0)
    direction = alert.get("direction", "ahead")

    distance_text = "1 meter" if round(distance_m) <= 1 else f"{round(distance_m)} meters"

    return f"Warning - {label} {direction}, {distance_text}"


# alert decision maker
def should_speak(alert: dict) -> bool:
    label = alert["label"]
    urgency = alert.get("urgency", "low")
    cooldown = URGENCY_COOLDOWN.get(urgency, ALERT_COOLDOWN)
    now = time.time()

    if now - _last_spoken.get(label, 0.0) >= cooldown:
        _last_spoken[label] = now
        return True

    return False


def vocalization_worker() -> None:
    # prioritizes the critical alerts and drops unecessary alerts
    while True:
        priority, timestamp, alert = _alert_queue.get()

        try:
            age = time.time() - timestamp

            if age > 3.0:
                log(f"Dropped stale alert: {alert['label']} ({age:.1f}s old)")
                continue

            if should_speak(alert):
                phrase = alert.get("_phrase") or build_phrase(alert)
                log(f"▶ {phrase} [{alert.get('urgency', '?')}]")
                speak_blocking(phrase)

        finally:
            _alert_queue.task_done()


def handle_client(conn: socket.socket, addr: tuple) -> None:
    log(f"Nano A connected from {addr[0]}:{addr[1]}")
    buffer = ""

    with conn:
        while True:
            try:
                chunk = conn.recv(1024).decode("utf-8")
            except OSError:
                break

            if not chunk:
                log(f"Nano A {addr[0]} disconnected")
                break

            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                try:
                    alert = json.loads(line)
                except json.JSONDecodeError as error:
                    log(f"Bad JSON from {addr[0]}: {error} - raw: {line!r}")
                    continue

                # startup msg
                if alert.get("label") == "__system__":
                    message = alert.get("message", "System ready.")
                    log(f"System message: {message}")

                    _alert_queue.put(
                        (
                            0,
                            time.time(),
                            {
                                "label": "__system__",
                                "distance_m": 0,
                                "direction": "",
                                "urgency": "critical",
                                "_phrase": message,
                            },
                        )
                    )
                    continue

                urgency = alert.get("urgency", "low")
                priority = URGENCY_RANK.get(urgency, 2)

                _alert_queue.put((priority, time.time(), alert))

                log(
                    f"← {alert.get('label', '?'):<14} "
                    f"{alert.get('direction', ''):<13} "
                    f"{alert.get('distance_m', '?'):>4} m "
                    f"[{urgency}]"
                )


def start_server(port: int) -> None:
    """
    Start the TCP server and listen for hazard alerts from Nano A.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((LISTEN_HOST, port))
        server.listen()

        log(f"Listening for Nano A on {LISTEN_HOST}:{port}...")

        while True:
            conn, addr = server.accept()

            threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True,
                name=f"client-{addr[0]}",
            ).start()

# test intended for only one nano
def self_test() -> None:

    log("Running self-test to test alerts without connection.")

    tests = [
        {
            "label": "chair",
            "distance_m": 1.2,
            "direction": "on your left",
            "urgency": "critical",
            "conf": 0.91,
        },
        {
            "label": "person",
            "distance_m": 2.5,
            "direction": "ahead",
            "urgency": "high",
            "conf": 0.87,
        },
        {
            "label": "bottle",
            "distance_m": 0.8,
            "direction": "on your right",
            "urgency": "critical",
            "conf": 0.78,
        },
        {
            "label": "car",
            "distance_m": 8.0,
            "direction": "ahead",
            "urgency": "low",
            "conf": 0.82,
        },
    ]

    for index, alert in enumerate(tests):
        priority = URGENCY_RANK.get(alert["urgency"], 2)
        _alert_queue.put((priority, time.time() + index * 0.01, alert))
        time.sleep(0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nano B - TTS audio receiver")

    parser.add_argument(
        "--port",
        type=int,
        default=LISTEN_PORT,
        help=f"TCP listen port (default {LISTEN_PORT})",
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Inject test alerts and speak them without Nano A",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    threading.Thread(
        target=vocalization_worker,
        daemon=True,
        name="vocalizer",
    ).start()

    # Startup message
    _alert_queue.put(
        (
            0,
            time.time(),
            {
                "label": "__system__",
                "distance_m": 0,
                "direction": "",
                "urgency": "critical",
                "_phrase": "Audio system ready.",
            },
        )
    )

    if args.self_test:
        self_test()
        _alert_queue.join()
        log("Self-test complete.")
        return

    start_server(args.port)


if __name__ == "__main__":
    main()