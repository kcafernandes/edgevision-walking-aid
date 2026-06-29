# EdgeVision Walking Aid

EdgeVision Walking Aid is a wearable assistive technology prototype designed to help visually impaired users receive real-time audio warnings about nearby hazards.

The system uses computer vision, depth estimation, and edge AI hardware to detect objects, estimate their distance, rank hazards, and send spoken warnings to the user.

## Tech Stack

- Python
- YOLOv8n
- Depth Anything V2
- TensorRT
- OpenCV
- PyTorch
- Jetson Orin Nano
- TCP sockets
- Text-to-speech audio alerts

## System Overview

EdgeVision uses two Jetson Orin Nano boards:

- **Vision Node:** captures camera frames, detects objects, estimates depth, and sends hazard alerts
- **Audio Node:** receives alerts, prioritizes urgent warnings, and speaks them to the user

Example alert:

```txt
Warning - car on your right, 3 meters
```

## Project Structure

```txt
edgevision-walking-aid/
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   └── edgevision/
│       ├── camera.py
│       ├── cloud_logger.py
│       ├── config.py
│       ├── depth.py
│       ├── detector.py
│       ├── hazard_logic.py
│       ├── networking.py
│       └── receiver.py
```

## Status

This repository is currently being organized from a working prototype into a clean, professional project structure.

## My Role

I worked on the edge AI pipeline, including object detection, depth estimation, hazard ranking, TCP communication between Jetson boards, and audio warning logic.

## Disclaimer

This is a student prototype and portfolio project. It is not a certified medical or safety device.
