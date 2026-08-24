import cv2
import mediapipe as mp
import math
import time

from collections import deque


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=2,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


# ============================================================
# CAMERA
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("Professional face movement monitor started.")
print("Press Q to close.")


# ============================================================
# SETTINGS
# ============================================================

YAW_THRESHOLD = 15

PITCH_UP_THRESHOLD = 8

PITCH_DOWN_THRESHOLD = 7


# Number of values used for smoothing

SMOOTHING_FRAMES = 7


# Movement must remain for this many frames

CONFIRMATION_FRAMES = 8


# Seconds before the same movement
# can trigger again

COOLDOWN_SECONDS = 2.0


# ============================================================
# VARIABLES
# ============================================================

timestamp = 0


baseline_yaw = None

baseline_pitch = None


yaw_history = deque(
    maxlen=SMOOTHING_FRAMES
)

pitch_history = deque(
    maxlen=SMOOTHING_FRAMES
)


candidate_movement = "NORMAL"

candidate_count = 0


confirmed_movement = "NORMAL"

last_event_time = 0


# ============================================================
# HEAD POSE CALCULATION
# ============================================================

def calculate_head_pose(landmarks):

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

    face_height = (
        mouth_center_y -
        eye_center_y
    )


    if abs(face_height) < 0.0001:

        return yaw, 0.0


    vertical_offset = (
        nose.y -
        eye_center_y
    )


    pitch = (
        vertical_offset /
        abs(face_height)
    ) * 30


    return yaw, pitch


# ============================================================
# CLASSIFY RAW MOVEMENT
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
# MOVEMENT CONFIRMATION
# ============================================================

def confirm_movement(current_movement):

    global candidate_movement
    global candidate_count
    global confirmed_movement
    global last_event_time


    current_time = time.time()


    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    if current_movement == "NORMAL":

        candidate_movement = "NORMAL"

        candidate_count = 0

        confirmed_movement = "NORMAL"

        return "NORMAL"


    # --------------------------------------------------------
    # Same movement continues
    # --------------------------------------------------------

    if current_movement == candidate_movement:

        candidate_count += 1

    else:

        candidate_movement = current_movement

        candidate_count = 1


    # --------------------------------------------------------
    # Check confirmation
    # --------------------------------------------------------

    if candidate_count >= CONFIRMATION_FRAMES:

        if (
            current_movement != confirmed_movement
            and
            current_time - last_event_time
            >= COOLDOWN_SECONDS
        ):

            confirmed_movement = (
                current_movement
            )

            last_event_time = current_time

            print(
                f"[CONFIRMED] {current_movement}"
            )

            return current_movement


    return confirmed_movement


# ============================================================
# MAIN LOOP
# ============================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        success, frame = camera.read()


        if not success:

            print(
                "ERROR: Could not read camera."
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
        # Convert BGR -> RGB
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


        timestamp += 33


        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )


        face_count = len(
            result.face_landmarks
        )


        # ====================================================
        # NO FACE
        # ====================================================

        if face_count == 0:

            status = "NO FACE"

            yaw = 0

            pitch = 0

            # Reset movement confirmation

            candidate_movement = "NORMAL"

            candidate_count = 0

            confirmed_movement = "NO FACE"


        # ====================================================
        # MULTIPLE FACES
        # ====================================================

        elif face_count > 1:

            status = "MULTIPLE FACES"

            yaw = 0

            pitch = 0

            candidate_movement = "NORMAL"

            candidate_count = 0

            confirmed_movement = (
                "MULTIPLE FACES"
            )


        # ====================================================
        # ONE FACE
        # ====================================================

        else:

            landmarks = (
                result.face_landmarks[0]
            )


            yaw, pitch = (
                calculate_head_pose(
                    landmarks
                )
            )


            # ------------------------------------------------
            # Establish baseline
            # ------------------------------------------------

            if baseline_yaw is None:

                baseline_yaw = yaw

                baseline_pitch = pitch


            # ------------------------------------------------
            # Add to smoothing history
            # ------------------------------------------------

            yaw_history.append(yaw)

            pitch_history.append(pitch)


            # ------------------------------------------------
            # Calculate smoothed values
            # ------------------------------------------------

            smooth_yaw = (
                sum(yaw_history) /
                len(yaw_history)
            )


            smooth_pitch = (
                sum(pitch_history) /
                len(pitch_history)
            )


            # ------------------------------------------------
            # Classify movement
            # ------------------------------------------------

            raw_movement = classify_movement(

                smooth_yaw,

                smooth_pitch,

                baseline_yaw,

                baseline_pitch
            )


            # ------------------------------------------------
            # Confirm movement
            # ------------------------------------------------

            result_movement = (
                confirm_movement(
                    raw_movement
                )
            )


            # ------------------------------------------------
            # Determine display status
            # ------------------------------------------------

            if result_movement != "NORMAL":

                status = result_movement

            else:

                status = raw_movement


            # ------------------------------------------------
            # Draw landmarks
            # ------------------------------------------------

            height, width, _ = (
                frame.shape
            )


            for landmark in landmarks:

                x = int(
                    landmark.x * width
                )

                y = int(
                    landmark.y * height
                )


                cv2.circle(

                    frame,

                    (x, y),

                    1,

                    (0, 255, 0),

                    -1
                )


            yaw = smooth_yaw

            pitch = smooth_pitch


        # ====================================================
        # DISPLAY COLOR
        # ====================================================

        if status == "NORMAL":

            text_color = (
                0,
                255,
                0
            )

        else:

            text_color = (
                0,
                0,
                255
            )


        # ====================================================
        # STATUS
        # ====================================================

        cv2.putText(

            frame,

            f"STATUS: {status}",

            (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            text_color,

            2
        )


        # ====================================================
        # FACE COUNT
        # ====================================================

        cv2.putText(

            frame,

            f"Faces: {face_count}",

            (20, 75),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        # ====================================================
        # YAW
        # ====================================================

        cv2.putText(

            frame,

            f"Yaw: {yaw:.2f}",

            (20, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        # ====================================================
        # PITCH
        # ====================================================

        cv2.putText(

            frame,

            f"Pitch: {pitch:.2f}",

            (20, 135),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        # ====================================================
        # CONFIRMATION PROGRESS
        # ====================================================

        if candidate_movement != "NORMAL":

            progress = min(
                candidate_count,
                CONFIRMATION_FRAMES
            )

            cv2.putText(

                frame,

                f"Confirming: "
                f"{candidate_movement} "
                f"{progress}/"
                f"{CONFIRMATION_FRAMES}",

                (20, 165),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 255),

                2
            )


        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(

            "Professional Face Movement Monitor",

            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()