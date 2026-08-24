# Online Assessment Face Movement Monitor

A Python-based face movement monitoring system designed for online assessments.

The system uses MediaPipe and OpenCV to monitor facial/head movements during an assessment and capture evidence when significant movements are detected.

It can detect:

- Looking Left
- Looking Right
- Looking Up
- Looking Down
- No Face
- Multiple Faces

The system also maintains session-based event logs and stores captured evidence images for later review.


## Project Structure

```text
face_movement_moniter/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── models/
│   └── face_landmarker.task
│
├── evidence/
│   ├── looking_left/
│   ├── looking_right/
│   ├── looking_up/
│   ├── looking_down/
│   ├── no_face/
│   └── multiple_faces/
│
├── logs/
│   └── events.csv
│
└── venv/
```


## Installation

### 1. Clone or download the project

Place the project folder on your computer.

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python main.py
```

For Git Bash, if MediaPipe diagnostic messages appear, you can run:

```bash
python main.py 2>/dev/null
```



## Configuration

The main detection settings are stored in `config.py`.

```python
YAW_THRESHOLD = 15

PITCH_UP_THRESHOLD = 0.7
PITCH_DOWN_THRESHOLD = 0.7

SMOOTHING_FRAMES = 7

CONFIRMATION_FRAMES = 8

COOLDOWN_SECONDS = 2.0
```

### Settings

- `YAW_THRESHOLD` — sensitivity for left/right head movement.
- `PITCH_UP_THRESHOLD` — sensitivity for upward head movement.
- `PITCH_DOWN_THRESHOLD` — sensitivity for downward head movement.
- `SMOOTHING_FRAMES` — number of frames used to smooth movement data.
- `CONFIRMATION_FRAMES` — continuous frames required before a movement is confirmed.
- `COOLDOWN_SECONDS` — minimum time between evidence captures.




## Evidence and Logs

### Evidence

Captured evidence images are stored in the `evidence/` directory.

Evidence is organized by event type:

- `looking_left/`
- `looking_right/`
- `looking_up/`
- `looking_down/`
- `no_face/`
- `multiple_faces/`

Each captured image contains a timestamp and unique event ID in its filename.

### Event Log

Assessment events are recorded in:

```text
logs/events.csv
```

The CSV contains:

- Session ID
- Event ID
- Timestamp
- Event type
- Yaw
- Pitch
- Event duration
- Evidence image path



## System Workflow

1. Start the application and initialize the camera.
2. Load the face landmark model.
3. Perform calibration while the user looks straight.
4. Detect the face and facial landmarks.
5. Calculate head pose using yaw and pitch.
6. Analyze eye gaze.
7. Smooth movement values to reduce noise.
8. Require consecutive frames before confirming a movement.
9. Capture evidence when a movement is confirmed.
10. Record the event in `logs/events.csv`.
11. Continue monitoring until the user presses `Q`.
12. Display the session summary when monitoring stops.



## Features

- Real-time face detection using MediaPipe.
- Head movement detection using yaw and pitch.
- Eye gaze detection.
- Face absence detection.
- Multiple-face detection.
- Movement smoothing to reduce false detections.
- Consecutive-frame confirmation before flagging movement.
- Evidence image capture for flagged events.
- Session-based event tracking.
- CSV event logging.
- Session summary after monitoring ends.
- Configurable detection thresholds.