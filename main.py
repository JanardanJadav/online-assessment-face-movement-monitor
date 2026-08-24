import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import mediapipe as mp
import math
import time
import csv

from config import (
    YAW_THRESHOLD,
    PITCH_UP_THRESHOLD,
    PITCH_DOWN_THRESHOLD,
    SMOOTHING_FRAMES,
    CONFIRMATION_FRAMES,
    COOLDOWN_SECONDS
)

from pathlib import Path
from collections import deque
from datetime import datetime


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

EVIDENCE_DIR = BASE_DIR / "evidence"

LOG_DIR = BASE_DIR / "logs"

LOG_FILE = LOG_DIR / "events.csv"


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

EVIDENCE_FOLDERS = {
    "LOOKING LEFT": "looking_left",
    "LOOKING RIGHT": "looking_right",
    "LOOKING UP": "looking_up",
    "LOOKING DOWN": "looking_down",
    "NO FACE": "no_face",
    "MULTIPLE FACES": "multiple_faces"
}

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

for folder in EVIDENCE_FOLDERS.values():

    (
        EVIDENCE_DIR / folder
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# CSV LOG INITIALIZATION
# ============================================================

if not LOG_FILE.exists():

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "session_id",
            "event_id",
            "timestamp",
            "event",
            "yaw",
            "pitch",
            "duration_seconds",
            "image_path"
        ])

# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = (
    mp.tasks.vision.FaceLandmarker
)

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)

options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=str(
            MODEL_PATH
        )
    ),

    running_mode=RunningMode.VIDEO,

    # Maximum two faces
    num_faces=2,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

camera = cv2.VideoCapture(
    CAMERA_INDEX
)

camera.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

camera.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

if not camera.isOpened():

    print(
        "ERROR: Camera could not be opened."
    )

    exit()


# ============================================================
# BASELINE CALIBRATION
# ============================================================

CALIBRATION_FRAMES = 30

calibration_yaw = []
calibration_pitch = []
baseline_yaw = None
baseline_pitch = None
calibration_eye_x = []
calibration_eye_y = []
baseline_eye_x = None
baseline_eye_y = None

# ============================================================
# SMOOTHING
# ============================================================

yaw_history = deque(
    maxlen=SMOOTHING_FRAMES
)

pitch_history = deque(
    maxlen=SMOOTHING_FRAMES
)

# ============================================================
# MOVEMENT STATE
# ============================================================

candidate_movement = "NORMAL"

candidate_count = 0

confirmed_movement = "NORMAL"

# ============================================================
# SESSION / EVENT STATE
# ============================================================

# Unique ID for this assessment session

SESSION_ID = (
    f"SESSION-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

event_counter = 0
last_event_time = 0
# last_event_type = None
event_start_time = None
active_event_id = None
active_event_type = None
normal_start_time = None

head_event_start_time = None
head_event_id = None
head_event_type = None

eye_event_start_time = None
eye_event_id = None
eye_event_type = None
eye_candidate = "EYES NORMAL"
eye_candidate_count = 0

# ============================================================
# DISPLAY STATE
# ============================================================

last_captured_event = "NONE"
last_special_state = "NORMAL"
last_capture_display_time = 0

# ============================================================
# TIMESTAMP
# ============================================================

video_timestamp = 0

# ============================================================
# HEAD POSE CALCULATION
# ============================================================

def calculate_head_pose(landmarks):

    """
    Estimate approximate yaw and pitch
    from MediaPipe facial landmarks.
    """

    nose = landmarks[1]

    left_eye = landmarks[33]
    right_eye = landmarks[263]

    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # --------------------------------------------------------
    # Eye center
    # --------------------------------------------------------

    eye_center_x = (
        left_eye.x +
        right_eye.x
    ) / 2

    eye_center_y = (
        left_eye.y +
        right_eye.y
    ) / 2

    # --------------------------------------------------------
    # Mouth center
    # --------------------------------------------------------

    mouth_center_y = (
        left_mouth.y +
        right_mouth.y
    ) / 2

    # --------------------------------------------------------
    # Eye distance
    # --------------------------------------------------------

    eye_distance = math.sqrt(

        (
            right_eye.x -
            left_eye.x
        ) ** 2

        +

        (
            right_eye.y -
            left_eye.y
        ) ** 2
    )


    if eye_distance < 0.0001:
        return 0.0, 0.0


    # ========================================================
    # YAW
    # ========================================================

    horizontal_offset = (
        nose.x -
        eye_center_x
    )

    yaw = (
        horizontal_offset /
        eye_distance
    ) * 60

    # ========================================================
    # PITCH
    # ========================================================
    # Use eye distance as a stable reference instead of
    # mouth-to-eye face height.

    face_vertical = (
    mouth_center_y -
    eye_center_y
    )

    if abs(face_vertical) < 0.0001:
        return yaw, 0.0

    vertical_offset = (
        nose.y -
        eye_center_y
    )

    pitch = (
        vertical_offset /
        abs(face_vertical)
    ) * 30

    return yaw, pitch

# ============================================================
# EYE GAZE CALCULATION
# ============================================================

def calculate_eye_gaze(
        landmarks,
        baseline_eye_x,
        baseline_eye_y):

    # ========================================================
    # IRIS CENTERS
    # ========================================================

    left_iris = landmarks[468]
    right_iris = landmarks[473]

    # ========================================================
    # HORIZONTAL EYE POSITION
    # ========================================================

    eye_center_x = (
        left_iris.x +
        right_iris.x
    ) / 2

    eye_difference_x = (
        eye_center_x -
        baseline_eye_x
    )

    # ========================================================
    # LEFT EYE VERTICAL POSITION
    # ========================================================

    left_upper = landmarks[159]
    left_lower = landmarks[145]

    left_eye_height = (
        left_lower.y -
        left_upper.y
    )

    if abs(left_eye_height) < 0.0001:
        left_eye_ratio = 0.5

    else:

        left_eye_ratio = (
            left_iris.y -
            left_upper.y
        ) / left_eye_height

    # ========================================================
    # RIGHT EYE VERTICAL POSITION
    # ========================================================

    right_upper = landmarks[386]
    right_lower = landmarks[374]

    right_eye_height = (
        right_lower.y -
        right_upper.y
    )

    if abs(right_eye_height) < 0.0001:
        right_eye_ratio = 0.5

    else:

        right_eye_ratio = (
            right_iris.y -
            right_upper.y
        ) / right_eye_height

    # ========================================================
    # AVERAGE BOTH EYES
    # ========================================================

    eye_center_y = (
        left_eye_ratio +
        right_eye_ratio
    ) / 2

    eye_difference_y = (
        eye_center_y -
        baseline_eye_y
    )

    # ========================================================
    # EYE LEFT / RIGHT
    # ========================================================

    if eye_difference_x < -0.015:

        return "EYES LOOKING LEFT"

    if eye_difference_x > 0.015:

        return "EYES LOOKING RIGHT"

    # ========================================================
    # EYE UP / DOWN
    # ========================================================

    if eye_difference_y < -0.05:

        return "EYES LOOKING DOWN"

    if eye_difference_y > 0.03:

        return "EYES LOOKING UP"

    return "EYES NORMAL"

# ============================================================
# MOVEMENT CLASSIFICATION
# ============================================================

def classify_movement(
    yaw,
    pitch,
    baseline_yaw,
    baseline_pitch
):

    relative_yaw = (
        yaw -
        baseline_yaw
    )

    relative_pitch = (
        pitch -
        baseline_pitch
    )

    # --------------------------------------------------------
    # LEFT
    # --------------------------------------------------------

    if relative_yaw < -YAW_THRESHOLD:

        return "LOOKING LEFT"

    # --------------------------------------------------------
    # RIGHT
    # --------------------------------------------------------

    if relative_yaw > YAW_THRESHOLD:

        return "LOOKING RIGHT"

    # --------------------------------------------------------
    # UP
    # --------------------------------------------------------

    if relative_pitch < -PITCH_UP_THRESHOLD:

        return "LOOKING UP"

    # --------------------------------------------------------
    # DOWN
    # --------------------------------------------------------

    if relative_pitch > PITCH_DOWN_THRESHOLD:

        return "LOOKING DOWN"

    return "NORMAL"

# ============================================================
# GENERATE EVENT ID
# ============================================================

def generate_event_id():

    global event_counter

    event_counter += 1

    return (
        f"EVT-{event_counter:05d}"
    )

def update_event_duration(
    event_id,
    duration_seconds
):

    rows = []

    if not LOG_FILE.exists():
        return

    with open(
        LOG_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        for row in reader:

            if (
                row["session_id"] == SESSION_ID
                and
                row["event_id"] == event_id
            ):

                row["duration_seconds"] = (
                    f"{duration_seconds:.2f}"
                )

            rows.append(row)


    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

# ============================================================
# SESSION SUMMARY
# ============================================================

def print_session_summary():

    if not LOG_FILE.exists():
        return

    event_counts = {
        "LOOKING LEFT": 0,
        "LOOKING RIGHT": 0,
        "LOOKING UP": 0,
        "LOOKING DOWN": 0,
        "NO FACE": 0,
        "MULTIPLE FACES": 0
    }

    total_events = 0

    with open(
        LOG_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["session_id"] != SESSION_ID:
                continue

            event = row["event"]

            if event in event_counts:

                event_counts[event] += 1
                total_events += 1

    print()
    print("=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    print(f"Session ID: {SESSION_ID}")
    print()

    for event, count in event_counts.items():

        print(
            f"{event:<18}: {count} event(s)"
        )

    print("-" * 60)

    print(
        f"Total flagged events : {total_events}"
    )

    print("=" * 60)

# ============================================================
# CAPTURE EVIDENCE
# ============================================================

def capture_evidence(
    frame,
    event_type,
    yaw,
    pitch
):

    global last_event_time
    # global last_event_type
    global last_captured_event
    global last_capture_display_time

    current_time = time.time()

    # --------------------------------------------------------
    # Cooldown protection
    # --------------------------------------------------------

    if (
        current_time -
        last_event_time
        <
        COOLDOWN_SECONDS
    ):

        return False

    event_id = generate_event_id()

    now = datetime.now()

    timestamp_string = (
        now.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    filename = (

        f"{now.strftime('%Y%m%d_%H%M%S')}"
        f"_{event_id}.jpg"

    )

    folder_name = (
        EVIDENCE_FOLDERS[
            event_type
        ]
    )

    image_directory = (
        EVIDENCE_DIR /
        folder_name
    )

    image_path = (
        image_directory /
        filename
    )

    # ========================================================
    # CREATE EVIDENCE IMAGE
    # ========================================================

    evidence_frame = frame.copy()

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    cv2.rectangle(

        evidence_frame,

        (0, 0),

        (
            evidence_frame.shape[1],
            120
        ),

        (20, 20, 20),

        -1
    )

    cv2.putText(

        evidence_frame,

        f"EVENT: {event_type}",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 0, 255),

        2
    )

    cv2.putText(

        evidence_frame,

        f"ID: {event_id}",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )

    cv2.putText(

        evidence_frame,

        timestamp_string,

        (20, 95),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    success = cv2.imwrite(

        str(image_path),

        evidence_frame
    )

    if not success:

        print(
            "[ERROR] Could not save evidence image."
        )

        return False

    # ========================================================
    # WRITE CSV LOG
    # ========================================================

    with open(

        LOG_FILE,

        "a",

        newline="",

        encoding="utf-8"

    ) as file:

        writer = csv.writer(file)

        writer.writerow([

            SESSION_ID,

            event_id,

            timestamp_string,

            event_type,

            round(yaw, 2),

            round(pitch, 2),

            "",

            str(image_path)

        ])

    # ========================================================
    # UPDATE STATE
    # ========================================================

    last_event_time = current_time
    # last_event_type = event_type
    last_captured_event = event_type
    last_capture_display_time = current_time

    # ========================================================
    # TERMINAL OUTPUT
    # ========================================================

    print(
    f"[EVIDENCE CAPTURED] {event_type}"
    )

    return event_id

# ============================================================
# DRAW LANDMARKS
# ============================================================

def draw_landmarks(
    frame,
    landmarks
):

    height, width, _ = (
        frame.shape
    )

    for landmark in landmarks:

        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )

        cv2.circle(

            frame,

            (x, y),

            1,

            (0, 255, 0),

            -1
        )

# ============================================================
# DRAW STATUS
# ============================================================

def draw_status(

    frame,

    status,

    face_count,

    yaw,

    pitch

):

    # --------------------------------------------------------
    # Status color
    # --------------------------------------------------------

    if status == "NORMAL":

        status_color = (
            0,
            255,
            0
        )

    elif status in (
        "NO FACE",
        "MULTIPLE FACES"
    ):

        status_color = (
            0,
            0,
            255
        )
    else:
        status_color = (
            0,
            165,
            255
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"STATUS: {status}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.85,

        status_color,

        2
    )

    # --------------------------------------------------------
    # Face count
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Faces: {face_count}",

        (20, 70),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )

    # --------------------------------------------------------
    # Yaw
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Yaw: {yaw:.2f}",

        (20, 100),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )

    # --------------------------------------------------------
    # Pitch
    # --------------------------------------------------------

    cv2.putText(

        frame,

        f"Pitch: {pitch:.2f}",

        (20, 130),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        (255, 255, 255),

        2
    )

# ============================================================
# DRAW CALIBRATION
# ============================================================

def draw_calibration(

    frame,

    current,

    total

):

    cv2.putText(

        frame,

        "CALIBRATING - LOOK STRAIGHT",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.8,

        (0, 255, 255),

        2
    )

    cv2.putText(

        frame,

        f"Progress: {current}/{total}",

        (20, 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2
    )

# ============================================================
# MAIN APPLICATION
# ============================================================

print()

print("=" * 60)

print(
    "ONLINE ASSESSMENT FACE MOVEMENT MONITOR"
)

print(
    f"Session ID: {SESSION_ID}"
)

print("=" * 60)

print()

print(
    "Look straight during calibration."
)

print(
    "Press Q to quit."
)

print()

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # CAMERA FRAME
        # ====================================================

        success, frame = camera.read()

        if not success:
            print(
                "ERROR: Could not read camera frame."
            )
            break

        # ----------------------------------------------------
        # Mirror camera
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # ----------------------------------------------------
        # RGB conversion
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        video_timestamp += 33

        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(
                mp_image,
                video_timestamp
            )
        )

        face_count = len(
            result.face_landmarks
        )

        # ====================================================
        # CALIBRATION
        # ====================================================

        if (

            baseline_yaw is None
            and
            face_count == 1

        ):

            landmarks = (
                result.face_landmarks[0]
            )

            # ------------------------------------------------
            # EYE GAZE
            # ------------------------------------------------

            yaw, pitch = (
                calculate_head_pose(
                    landmarks
                )
            )

            eye_x = (
                landmarks[468].x +
                landmarks[473].x
            ) / 2

            calibration_eye_x.append(
                eye_x
            )

            # ------------------------------------------------
            # EYE VERTICAL CALIBRATION
            # ------------------------------------------------

            left_iris = landmarks[468]
            right_iris = landmarks[473]

            left_upper = landmarks[159]
            left_lower = landmarks[145]

            right_upper = landmarks[386]
            right_lower = landmarks[374]


            left_eye_height = (
                left_lower.y -
                left_upper.y
            )

            right_eye_height = (
                right_lower.y -
                right_upper.y
            )

            if (
                abs(left_eye_height) > 0.0001
                and
                abs(right_eye_height) > 0.0001
            ):

                left_eye_ratio = (
                    left_iris.y -
                    left_upper.y
                ) / left_eye_height

                right_eye_ratio = (
                    right_iris.y -
                    right_upper.y
                ) / right_eye_height

                eye_y = (
                    left_eye_ratio +
                    right_eye_ratio
                ) / 2

                calibration_eye_y.append(
                    eye_y
                )

            calibration_yaw.append(
                yaw
            )

            calibration_pitch.append(
                pitch
            )

            draw_landmarks(
                frame,
                landmarks
            )

            calibration_count = (
                len(calibration_yaw)
            )

            draw_calibration(

                frame,

                calibration_count,

                CALIBRATION_FRAMES
            )


            if (

                calibration_count
                >=
                CALIBRATION_FRAMES

            ):


                baseline_yaw = (
                    sum(calibration_yaw)
                    /
                    len(calibration_yaw)
                )


                baseline_pitch = (
                    sum(calibration_pitch)
                    /
                    len(calibration_pitch)
                )

                baseline_eye_x = (
                    sum(calibration_eye_x) /
                    len(calibration_eye_x)
                )

                baseline_eye_y = (
                sum(calibration_eye_y) /
                len(calibration_eye_y)
                )

                yaw_history.clear()
                pitch_history.clear()

                print()

                print(
                    "[CALIBRATION COMPLETE]"
                )

                print()


            cv2.imshow(

                "Professional Face Movement Monitor",

                frame
            )


            if (
                cv2.waitKey(1)
                &
                0xFF
            ) == ord("q"):

                break

            continue

    # ====================================================
    # NORMAL MONITORING
    # ====================================================
    # Default values for every frame.
    # This prevents undefined-variable errors.

        status = "NORMAL"
        yaw = 0.0
        pitch = 0.0


        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if face_count == 0:

            status = "NO FACE"

            candidate_movement = "NORMAL"
            candidate_count = 0
            confirmed_movement = "NO FACE"

            # Capture only when entering NO FACE state

            if last_special_state != "NO FACE":

                event_start_time = time.time()

                active_event_id = capture_evidence(
                    frame,
                    "NO FACE",
                    yaw,
                    pitch
                )

                active_event_type = "NO FACE"

                last_special_state = "NO FACE"

        # ----------------------------------------------------
        # MULTIPLE FACES
        # ----------------------------------------------------

        elif face_count > 1:

            status = "MULTIPLE FACES"

            candidate_movement = "NORMAL"
            candidate_count = 0
            confirmed_movement = "MULTIPLE FACES"


            # Capture only when entering
            # MULTIPLE FACES state

            if last_special_state != "MULTIPLE FACES":

                event_start_time = time.time()

                active_event_id = capture_evidence(
                    frame,
                    "MULTIPLE FACES",
                    yaw,
                    pitch
                )

                active_event_type = "MULTIPLE FACES"
                last_special_state = "MULTIPLE FACES"

        # ----------------------------------------------------
        # EXACTLY ONE FACE
        # ----------------------------------------------------

        elif face_count == 1:

            # ------------------------------------------------
            # Finish NO FACE event
            # ------------------------------------------------

            if (
                active_event_type == "NO FACE"
                and
                event_start_time is not None
                and
                active_event_id is not None
            ):

                event_duration = (
                    time.time() -
                    event_start_time
                )

                update_event_duration(
                    active_event_id,
                    event_duration
                )

                event_start_time = None
                active_event_id = None
                active_event_type = None

            # ------------------------------------------------
            # Finish MULTIPLE FACES event
            # ------------------------------------------------

            if (
                active_event_type == "MULTIPLE FACES"
                and
                event_start_time is not None
                and
                active_event_id is not None
            ):

                event_duration = (
                    time.time() -
                    event_start_time
                )

                update_event_duration(
                    active_event_id,
                    event_duration
                )

                event_start_time = None
                active_event_id = None
                active_event_type = None

            # A valid face has returned.
            # This resets NO FACE / MULTIPLE FACE state.
    
            last_special_state = "NORMAL"
            landmarks = result.face_landmarks[0]

            # ------------------------------------------------
            # Calculate head pose FIRST
            # ------------------------------------------------

            yaw, pitch = calculate_head_pose(
                landmarks
            )

            # ------------------------------------------------
            # EYE GAZE
            # ------------------------------------------------

            eye_gaze = calculate_eye_gaze(
                landmarks,
                baseline_eye_x,
                baseline_eye_y
            )

            # ------------------------------------------------
            # EYE GAZE EVENT DETECTION
            # ------------------------------------------------

            if eye_gaze == "EYES NORMAL":

                eye_candidate = "EYES NORMAL"
                eye_candidate_count = 0

                # Finish active eye event

                if (
                    eye_event_type in (
                        "EYES LOOKING LEFT",
                        "EYES LOOKING RIGHT",
                        "EYES LOOKING UP",
                        "EYES LOOKING DOWN"
                    )
                    and
                    eye_event_start_time is not None
                    and
                    eye_event_id is not None
                ):

                    event_duration = (
                        time.time() -
                        eye_event_start_time
                    )

                    update_event_duration(
                        eye_event_id,
                        event_duration
                    )

                    eye_event_start_time = None
                    eye_event_id = None
                    eye_event_type = None


            else:

                # Same eye movement continues

                if eye_gaze == eye_candidate:

                    eye_candidate_count += 1

                # New eye movement

                else:

                    eye_candidate = eye_gaze
                    eye_candidate_count = 1


                # ------------------------------------------------
                # CONFIRM EYE MOVEMENT
                # ------------------------------------------------

                if eye_candidate_count >= CONFIRMATION_FRAMES:

                    if eye_event_type != eye_candidate:

                        eye_event = eye_candidate.replace(
                            "EYES ",
                            ""
                        )

                        eye_event_start_time = time.time()

                        eye_event_id = capture_evidence(
                            frame,
                            eye_event,
                            yaw,
                            pitch
                        )

                        eye_event_type = eye_candidate

            # ------------------------------------------------
            # Smoothing
            # ------------------------------------------------

            yaw_history.append(yaw)
            pitch_history.append(pitch)

            smooth_yaw = (
                sum(yaw_history)
                /
                len(yaw_history)
            )


            smooth_pitch = (
                sum(pitch_history)
                /
                len(pitch_history)
            )


            yaw = smooth_yaw
            pitch = smooth_pitch

        
            # ------------------------------------------------
            # Classify movement
            # ------------------------------------------------

            raw_movement = classify_movement(
                yaw,
                pitch,
                baseline_yaw,
                baseline_pitch
            )

            # ------------------------------------------------
            # NORMAL POSITION
            # ------------------------------------------------

            if raw_movement == "NORMAL":

                status = "NORMAL"

                candidate_movement = "NORMAL"

                candidate_count = 0

                    # Start NORMAL grace-period timer

                if normal_start_time is None:

                    normal_start_time = time.time()

                # ------------------------------------------------
                # Finish active movement event
                # ------------------------------------------------

                if (
                    head_event_type in (
                        "LOOKING LEFT",
                        "LOOKING RIGHT",
                        "LOOKING UP",
                        "LOOKING DOWN"
                    )
                    and
                    head_event_start_time is not None
                    and
                    head_event_id is not None
                    and
                    normal_start_time is not None
                    and
                    time.time() - normal_start_time >= 0.75
                ):

                    event_duration = (
                        time.time() -
                        head_event_start_time
                    )

                    update_event_duration(
                        head_event_id,
                        event_duration
                    )

                    # Reset active event

                    head_event_start_time = None

                    head_event_id = None

                    head_event_type = None

                    normal_start_time = None

                    confirmed_movement = "NORMAL"

                    # last_event_type = None

            # ------------------------------------------------
            # MOVEMENT DETECTED
            # ------------------------------------------------

            else:

                status = raw_movement

                normal_start_time = None

                # Same movement continues

                if raw_movement == candidate_movement:

                    candidate_count += 1

                # New movement

                else:

                    candidate_movement = raw_movement

                    candidate_count = 1

                # ------------------------------------------------
                # Confirm movement
                # ------------------------------------------------

                if candidate_count >= CONFIRMATION_FRAMES:

                    if head_event_type != raw_movement:

                        confirmed_movement = raw_movement

                        # --------------------------------------------
                        # Start event timer
                        # --------------------------------------------

                        head_event_start_time = time.time()

                        # --------------------------------------------
                        # Capture evidence
                        # --------------------------------------------

                        head_event_id = capture_evidence(

                            frame,

                            raw_movement,

                            yaw,

                            pitch

                        )

                        head_event_type = raw_movement

                # ------------------------------------------------
                # Confirmation display
                # ------------------------------------------------

                progress = min(
                    candidate_count,
                    CONFIRMATION_FRAMES
                )

                cv2.putText(
                    frame,

                    (
                        f"Confirming: "
                        f"{candidate_movement} "
                        f"{progress}/"
                        f"{CONFIRMATION_FRAMES}"
                    ),

                    (20, 165),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 255),

                    2
                )

            # ------------------------------------------------
            # Draw face landmarks
            # ------------------------------------------------

            draw_landmarks(
                frame,
                landmarks
            )
        # ====================================================
        # DISPLAY STATUS
        # ====================================================

        draw_status(

            frame,

            status,

            face_count,

            yaw,

            pitch
        )


        # ====================================================
        # LAST CAPTURE MESSAGE
        # ====================================================

        if (

            time.time()
            -
            last_capture_display_time
            <
            2

        ):

            cv2.putText(

                frame,

                (
                    f"EVIDENCE SAVED: "
                    f"{last_captured_event}"
                ),

                (20, 200),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0, 255, 0),

                2
            )

        # ====================================================
        # SHOW CAMERA
        # ====================================================

        cv2.imshow(

            "Professional Face Movement Monitor",

            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if (
            cv2.waitKey(1)
            &
            0xFF
        ) == ord("q"):

            # Finish active head movement event before exiting

            if (
                head_event_start_time is not None
                and
                head_event_id is not None
                and
                head_event_type is not None
            ):

                event_duration = (
                    time.time() -
                    head_event_start_time
                )

                update_event_duration(
                    head_event_id,
                    event_duration
                )

                head_event_start_time = None
                head_event_id = None
                head_event_type = None

            break
            

# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print()

print(
    "Monitoring stopped."
)

print_session_summary()