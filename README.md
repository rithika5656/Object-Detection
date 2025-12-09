# Object Detection - Contact Tracing

Real-time contamination detection using YOLOv8.

## Features
- Detects patients (persons) and objects (bottles, cups, glasses)
- Tracks when hands touch objects
- Marks objects as CONTAMINATED

## Setup
```bash
pip install ultralytics opencv-python numpy
```

## Run
```bash
python main.py
```
Press 'q' to quit.

## How it Works
1. **YOLOv8** detects persons and objects
2. **YOLOv8-Pose** detects hand positions
3. **IoU algorithm** detects hand-object contact
4. Objects turn RED when contaminated

## Files
- `main.py` - Main detection script
- `requirements.txt` - Dependencies
- `.gitignore` - Excludes model files (auto-download)
