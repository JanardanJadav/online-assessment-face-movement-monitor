# Online Assessment Face Movement Monitor

A real-time face and head movement monitoring system designed for online assessments.

The system detects significant facial and head movements during an assessment, monitors face presence, records monitoring events, and provides evidence for later review.

The project includes both a Python-based computer vision implementation and a browser-based monitoring application.

##  Live Demo

### Browser Monitoring Application

**Live Application:**

https://janardanjadav.github.io/online-assessment-face-movement-monitor/

> Camera permission is required to use the live monitoring features.

---

##  Features

- Real-time face detection
- Face landmark tracking
- Head movement detection
- Looking Left detection
- Looking Right detection
- Looking Up detection
- Looking Down detection
- No Face detection
- Multiple Face detection
- Movement smoothing
- Consecutive-frame movement confirmation
- Calibration before monitoring
- Real-time face count
- Live monitoring status
- Session-based event tracking
- Event log
- Evidence capture
- Browser-based monitoring dashboard
- GitHub Pages deployment
- Automated deployment using GitHub Actions

---

#  System Architecture

The project contains two monitoring implementations.

## 1. Python Monitoring Application

The original implementation uses Python with OpenCV and MediaPipe.

It performs real-time monitoring through the computer camera and analyzes facial landmarks and head movement.

### Technologies

- Python
- OpenCV
- MediaPipe
- Face Landmarker

### Main Components

- Face detection
- Head pose analysis
- Eye gaze analysis
- Movement smoothing
- Consecutive-frame confirmation
- Evidence capture
- Event tracking

---

## 2. Browser Monitoring Application

The browser version provides a professional real-time monitoring dashboard that runs directly in a web browser.

### Technologies

- React
- Vite
- JavaScript
- MediaPipe Tasks Vision
- HTML
- CSS

### Dashboard

The browser application provides:

- Live camera feed
- Monitoring status
- Current movement
- Face count
- Session ID
- Event log
- Evidence gallery
- Real-time face movement detection

---

#  Project Structure

```text
online-assessment-face-movement-monitor/
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── docs/
│
├── models/
│   └── face_landmarker.task
│
├── web_app/
│   └── index.html
│
├── react_app/
│   ├── public/
│   │   ├── models/
│   │   │   └── face_landmarker.task
│   │   │
│   │   └── mediapipe/
│   │       ├── vision_wasm_internal.js
│   │       ├── vision_wasm_internal.wasm
│   │       ├── vision_wasm_module_internal.js
│   │       ├── vision_wasm_module_internal.wasm
│   │       ├── vision_wasm_nosimd_internal.js
│   │       └── vision_wasm_nosimd_internal.wasm
│   │
│   ├── src/
│   │   ├── assets/
│   │   │   ├── hero.png
│   │   │   ├── react.svg
│   │   │   └── vite.svg
│   │   │
│   │   ├── components/
│   │   │   └── CameraPanel.jsx
│   │   │
│   │   ├── services/
│   │   │   └── mediaPipe.js
│   │   │
│   │   ├── styles/
│   │   │   └── index.css
│   │   │
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   └── README.md
│
├── camera_test.py
├── config.py
├── face_detection.py
├── main.py
├── requirements.txt
└── README.md
```

---

#  Detection System

The monitoring system analyzes facial landmarks to identify head movement.

The browser application detects:

| Detection | Description |
|---|---|
| `LOOKING LEFT` | User turns their head toward the left |
| `LOOKING RIGHT` | User turns their head toward the right |
| `LOOKING UP` | User moves their head upward |
| `LOOKING DOWN` | User moves their head downward |
| `NO FACE` | No face is detected |
| `MULTIPLE FACES` | More than one face is detected |
| `LOOKING STRAIGHT` | Face remains within the calibrated normal range |

---

#  Monitoring Workflow

```text
Start Camera
     │
     ▼
Initialize MediaPipe Face Landmarker
     │
     ▼
Calibrate User
     │
     ▼
Detect Face & Facial Landmarks
     │
     ▼
Calculate Face Position
     │
     ▼
Analyze Head Movement
     │
     ▼
Apply Movement Confirmation
     │
     ▼
Record Monitoring Event
     │
     ▼
Capture Evidence
     │
     ▼
Update Event Log
     │
     ▼
Continue Monitoring
     │
     ▼
Stop Monitoring
```

---

#  Calibration

Before active monitoring begins, the system performs calibration while the user is looking straight at the camera.

Calibration establishes a baseline face position.

Subsequent face-position changes are compared against this baseline to determine head movement.

This helps reduce false movement detections caused by differences in camera position and user positioning.

---

#  Browser Application

The browser application is located inside:

```text
react_app/
```

## Requirements

- Node.js
- Modern web browser
- Camera
- Camera permission

## Install Dependencies

Navigate to the browser application:

```bash
cd react_app
```

Install dependencies:

```bash
npm install
```

## Run Locally

Start the Vite development server:

```bash
npm run dev
```

The application will normally be available at:

```text
http://localhost:5173/
```

## Production Build

Create a production build:

```bash
npm run build
```

The production files are generated inside:

```text
react_app/dist/
```

---

#  Python Application

The original desktop monitoring implementation is based on Python, OpenCV, and MediaPipe.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/JanardanJadav/online-assessment-face-movement-monitor.git
```

### 2. Navigate to the project

```bash
cd online-assessment-face-movement-monitor
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python main.py
```

For Git Bash, if MediaPipe diagnostic messages appear:

```bash
python main.py 2>/dev/null
```

---

#  Configuration

The Python application's detection settings are stored in:

```text
config.py
```

Example configuration:

```python
YAW_THRESHOLD = 15

PITCH_UP_THRESHOLD = 0.7
PITCH_DOWN_THRESHOLD = 0.7

SMOOTHING_FRAMES = 7

CONFIRMATION_FRAMES = 8

COOLDOWN_SECONDS = 2.0
```

### Configuration Parameters

| Parameter | Description |
|---|---|
| `YAW_THRESHOLD` | Sensitivity for left/right head movement |
| `PITCH_UP_THRESHOLD` | Sensitivity for upward head movement |
| `PITCH_DOWN_THRESHOLD` | Sensitivity for downward head movement |
| `SMOOTHING_FRAMES` | Number of frames used to smooth movement data |
| `CONFIRMATION_FRAMES` | Consecutive frames required before confirming movement |
| `COOLDOWN_SECONDS` | Minimum time between evidence captures |

---

#  Evidence Capture

The monitoring system can capture evidence when a significant movement or monitoring event is detected.

The Python implementation organizes captured evidence by event type.

Typical event categories include:

```text
looking_left/
looking_right/
looking_up/
looking_down/
no_face/
multiple_faces/
```

The browser application also provides an evidence gallery within the monitoring dashboard.

---

#  Event Tracking

Monitoring events are associated with a session ID.

The browser application displays events in a structured event log containing information such as:

- Session ID
- Date
- Timestamp
- Event type

This allows monitoring activity to be associated with a specific assessment session.

---

#  Session Tracking

Each monitoring session receives a unique session identifier.

Example:

```text
20260905-XYZABD
```

The session ID is displayed in the monitoring dashboard and associated with recorded monitoring events.

This provides a simple way to distinguish events from different assessment sessions.

---

#  Deployment

The browser application is deployed using **GitHub Pages**.

Deployment is automated through **GitHub Actions**.

The deployment workflow:

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ▼
Install Dependencies
   │
   ▼
Build Vite Application
   │
   ▼
Generate Production Build
   │
   ▼
Deploy to GitHub Pages
```

### Live Application

https://janardanjadav.github.io/online-assessment-face-movement-monitor/

---

#  Technologies Used

## Computer Vision

- MediaPipe
- OpenCV
- Face Landmarker

## Backend / Desktop

- Python

## Frontend

- React
- JavaScript
- HTML
- CSS
- Vite
- MediaPipe Tasks Vision

## Deployment

- GitHub
- GitHub Actions
- GitHub Pages

---

#  Key Highlights

- Real-time computer vision monitoring
- Face landmark-based movement analysis
- Head movement detection using facial landmarks
- Face absence detection
- Multiple-face detection
- Calibration-based monitoring
- Movement smoothing
- Consecutive-frame confirmation
- Session-based event tracking
- Evidence capture
- Professional browser monitoring dashboard
- Client-side MediaPipe processing
- Automated GitHub Pages deployment

---

#  Live Application

Try the browser-based monitoring application:

**https://janardanjadav.github.io/online-assessment-face-movement-monitor/**

---

#  Author

**Janardan Jadav**

Computer Science Graduate

Interested in software development, computer vision, and real-time web applications.
