import { useEffect, useRef, useState } from "react";
import {
  initializeFaceLandmarker,
  detectFaceLandmarks,
} from "../services/mediaPipe";

function CameraPanel({ 
            setEvents: updateAppEvents,
            setEvidence: updateAppEvidence,
            sessionId,
            setMonitoringStatus,
            setCurrentMovement,
            setMonitoringFaceCount,
 }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const animationFrameRef = useRef(null);

  // --------------------------------------------------
  // CAMERA / DETECTOR STATE
  // --------------------------------------------------

  const [cameraState, setCameraState] = useState("off");
  const [errorMessage, setErrorMessage] = useState("");
  const [faceCount, setFaceCount] = useState(0);
  const [detectorState, setDetectorState] = useState("idle");
  const [events, setEvents] = useState([]);
  const eventCountRef = useRef(0);

  // --------------------------------------------------
  // CALIBRATION STATE
  // --------------------------------------------------

  const [calibrationState, setCalibrationState] =
    useState("not-calibrated");

  const calibrationSamplesRef = useRef([]);

  const baselineRef = useRef(null);

  const calibrationStateRef = useRef("not-calibrated");

// --------------------------------------------------
// HEAD MOVEMENT DETECTION
// --------------------------------------------------

  const movementStateRef = useRef("NORMAL");
  const movementCandidateRef = useRef("NORMAL");
  const movementFramesRef = useRef(0);

// Prevent repeated console logs for the same state
  const lastLoggedMovementRef = useRef(null);
  const lastLoggedFaceStatusRef = useRef(null);

  const MOVEMENT_THRESHOLD_X = 0.12;
  const MOVEMENT_THRESHOLD_Y = 0.12;
  const MOVEMENT_CONFIRMATION_FRAMES = 6;

  const CALIBRATION_SAMPLE_COUNT = 45;

  // --------------------------------------------------
  // GET FACE POSITION
  // --------------------------------------------------
  //
  // We use stable facial landmarks:
  //
  // 1   = nose
  // 234 = left side of face
  // 454 = right side of face
  // 10  = forehead
  // 152 = chin
  //
  // The values are normalized so that calibration
  // works with different camera resolutions.
  // --------------------------------------------------

  const getFacePosition = (landmarks) => {
    if (!landmarks || landmarks.length === 0) {
      return null;
    }

    const nose = landmarks[1];
    const leftFace = landmarks[234];
    const rightFace = landmarks[454];
    const forehead = landmarks[10];
    const chin = landmarks[152];

    if (
      !nose ||
      !leftFace ||
      !rightFace ||
      !forehead ||
      !chin
    ) {
      return null;
    }

    // Face center
    const faceCenterX =
      (leftFace.x + rightFace.x) / 2;

    const faceCenterY =
      (forehead.y + chin.y) / 2;

    // Face dimensions
    const faceWidth =
      Math.abs(rightFace.x - leftFace.x);

    const faceHeight =
      Math.abs(chin.y - forehead.y);

    if (
      faceWidth === 0 ||
      faceHeight === 0
    ) {
      return null;
    }

    // Normalized nose position relative to
    // the center of the face.
    const normalizedX =
      (nose.x - faceCenterX) / faceWidth;

    const normalizedY =
      (nose.y - faceCenterY) / faceHeight;

    return {
      x: normalizedX,
      y: normalizedY,
    };
  };

// --------------------------------------------------
// HEAD MOVEMENT CLASSIFICATION
// --------------------------------------------------

const detectHeadMovement = (position) => {
  if (!position || !baselineRef.current) {
    return "NORMAL";
  }

  const baseline = baselineRef.current;

  const deltaX = position.x - baseline.x;
  const deltaY = position.y - baseline.y;

  let movement = "NORMAL";

  // Horizontal movement
  if (deltaX > MOVEMENT_THRESHOLD_X) {
    movement = "LOOKING LEFT";
  } else if (deltaX < -MOVEMENT_THRESHOLD_X) {
    movement = "LOOKING RIGHT";
  }

  // Vertical movement
  if (deltaY > MOVEMENT_THRESHOLD_Y) {
    movement = "LOOKING DOWN";
  } else if (deltaY < -MOVEMENT_THRESHOLD_Y) {
    movement = "LOOKING UP";
  }

  // Confirmation / stability check
  if (movement === movementCandidateRef.current) {
    movementFramesRef.current += 1;
  } else {
    movementCandidateRef.current = movement;
    movementFramesRef.current = 1;
  }

  if (
    movementFramesRef.current >=
    MOVEMENT_CONFIRMATION_FRAMES
  ) {
    movementStateRef.current = movement;
  }

  return movementStateRef.current;
};

const addEvent = (type) => {
const now = new Date();

const newEvent = {
  id: Date.now(),
  sessionId,
  type,
  date: now.toLocaleDateString("en-GB"),
  timestamp: now.toLocaleTimeString(),
};

  eventCountRef.current += 1;

  setEvents((prev) => [...prev, newEvent]);

  if (updateAppEvents) {
    updateAppEvents((prev) => [...prev, newEvent]);
  }

  // Capture camera evidence
  const video = videoRef.current;

  if (
    updateAppEvidence &&
    video &&
    video.videoWidth > 0 &&
    video.videoHeight > 0
  ) {
    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (context) {
      context.drawImage(
        video,
        0,
        0,
        video.videoWidth,
        video.videoHeight
      );

      // Add evidence information to the captured image
      const captureDate = new Date();

      const dateText = captureDate.toLocaleDateString("en-GB");
      const timeText = captureDate.toLocaleTimeString();

      context.fillStyle = "#ffffff";
      context.font = "bold 18px Arial";

      const sessionLine = `SESSION: ${sessionId}`;
      const dateLine = `DATE: ${dateText}`;
      const timeLine = `TIME: ${timeText}`;
      const eventLine = `EVENT: ${type}`;

      const textWidth = Math.max(
        context.measureText(sessionLine).width,
        context.measureText(dateLine).width,
        context.measureText(timeLine).width,
        context.measureText(eventLine).width
      );

      const boxWidth = textWidth + 30;
      const boxHeight = 125;

      context.fillStyle = "rgba(0, 0, 0, 0.75)";
      context.fillRect(20, 20, boxWidth, boxHeight);

      context.fillStyle = "#ffffff";

      context.fillText(sessionLine, 35, 50);
      context.fillText(dateLine, 35, 77);
      context.fillText(timeLine, 35, 104);
      context.fillText(eventLine, 35, 131);

      const image = canvas.toDataURL("image/jpeg", 0.85);

      updateAppEvidence((prev) => [
        ...prev,
        {
          id: newEvent.id,
          sessionId: newEvent.sessionId,
          type: newEvent.type,
          timestamp: newEvent.timestamp,
          image,
        },
      ]);
    }
  }

  console.log("EVENT CREATED:", newEvent);
};

  // --------------------------------------------------
  // FACE DETECTION LOOP
  // --------------------------------------------------

  const detectFaces = () => {
    const video = videoRef.current;

    if (!video) {
      animationFrameRef.current =
        requestAnimationFrame(detectFaces);

      return;
    }

    if (
      video.readyState <
        HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      animationFrameRef.current =
        requestAnimationFrame(detectFaces);

      return;
    }

    try {
      const timestamp = performance.now();

      const result = detectFaceLandmarks(
        video,
        timestamp
      );

      if (result?.faceLandmarks) {

  const detectedFaces =
    result.faceLandmarks.length;

  setFaceCount(detectedFaces);
  setMonitoringFaceCount(detectedFaces);

// --------------------------------------------------
// FACE COUNT STATUS
// --------------------------------------------------

let faceStatus;

if (detectedFaces === 0) {
  faceStatus = "NO FACE";
} else if (detectedFaces === 1) {
  faceStatus = "ONE FACE";
} else {
  faceStatus = "MULTIPLE FACES";
}

if (
  calibrationStateRef.current === "calibrated" &&
  (faceStatus === "NO FACE" ||
    faceStatus === "MULTIPLE FACES")
) {
  setCurrentMovement(faceStatus);
}

if (faceStatus !== lastLoggedFaceStatusRef.current) {
  console.log("FACE STATUS:", faceStatus);

  lastLoggedFaceStatusRef.current = faceStatus;

  if (
    calibrationStateRef.current === "calibrated" &&
    (faceStatus === "NO FACE" ||
    faceStatus === "MULTIPLE FACES")
  ) {
    addEvent(faceStatus);
  }
}

  // --------------------------------------------------
  // CALIBRATION
  // --------------------------------------------------

  if (
    calibrationStateRef.current === "calibrating" &&
    detectedFaces === 1
  ) {
    const position =
      getFacePosition(
        result.faceLandmarks[0]
      );

    if (position) {
      calibrationSamplesRef.current.push(
        position
      );

      const sampleCount =
        calibrationSamplesRef.current.length;

      if (
        sampleCount >=
        CALIBRATION_SAMPLE_COUNT
      ) {
        const samples =
          calibrationSamplesRef.current;

        const averageX =
          samples.reduce(
            (sum, sample) =>
              sum + sample.x,
            0
          ) / samples.length;

        const averageY =
          samples.reduce(
            (sum, sample) =>
              sum + sample.y,
            0
          ) / samples.length;

        baselineRef.current = {
          x: averageX,
          y: averageY,
        };

        calibrationSamplesRef.current = [];

        calibrationStateRef.current = "calibrated";

        setCalibrationState(
          "calibrated"
        );

        console.log(
          "Calibration completed:",
          {
            x: averageX,
            y: averageY,
          }
        );
      }
    }
  }

// --------------------------------------------------
// HEAD MOVEMENT DETECTION
// --------------------------------------------------

if (
  calibrationStateRef.current === "calibrated" &&
  detectedFaces === 1
) {
  const position =
    getFacePosition(
      result.faceLandmarks[0]
    );

  if (position) {
    const movement =
      detectHeadMovement(position);

    setCurrentMovement(
      movement === "NORMAL"
        ? "LOOKING STRAIGHT"
        : movement
    );

    if (movement !== lastLoggedMovementRef.current) {
  console.log(
    "HEAD MOVEMENT:",
    movement
  );

  lastLoggedMovementRef.current = movement;

  if (movement !== "NORMAL") {
    addEvent(movement);
  }
}
  }
}

        // --------------------------------------------------
        // CALIBRATION
        // --------------------------------------------------

        if (
          calibrationStateRef.current === "calibrating" &&
          detectedFaces === 1
        ) {
          const position =
            getFacePosition(
              result.faceLandmarks[0]
            );

          if (position) {
            calibrationSamplesRef.current.push(
              position
            );

            const sampleCount =
              calibrationSamplesRef.current.length;

            if (
              sampleCount >=
              CALIBRATION_SAMPLE_COUNT
            ) {
              const samples =
                calibrationSamplesRef.current;

              const averageX =
                samples.reduce(
                  (sum, sample) =>
                    sum + sample.x,
                  0
                ) / samples.length;

              const averageY =
                samples.reduce(
                  (sum, sample) =>
                    sum + sample.y,
                  0
                ) / samples.length;

              baselineRef.current = {
                x: averageX,
                y: averageY,
              };

              calibrationSamplesRef.current = [];

              calibrationStateRef.current = "calibrated"

              setCalibrationState("calibrated");

              console.log(
                "Calibration completed:",
                {
                  x: averageX,
                  y: averageY,
                }
              );
            }
          }
        }

      } else {
        setFaceCount(0);
        setMonitoringFaceCount(0);
      }

    } catch (error) {
      console.error(
        "Face detection error:",
        error
      );
    }

    animationFrameRef.current =
      requestAnimationFrame(detectFaces);
  };

  // --------------------------------------------------
  // START FACE DETECTION
  // --------------------------------------------------

  const startFaceDetection = async () => {
    try {
      setDetectorState("loading");

      await initializeFaceLandmarker();

      console.log(
        "Face detection is ready."
      );

      setDetectorState("active");

      if (!animationFrameRef.current) {
        detectFaces();
      }

    } catch (error) {
      console.error(
        "MediaPipe initialization error:",
        error
      );

      setDetectorState("error");

      setErrorMessage(
        "Unable to initialize face detection. Please refresh and try again."
      );
    }
  };

  // --------------------------------------------------
  // START CAMERA
  // --------------------------------------------------

  const startCamera = async () => {
    try {
      setErrorMessage("");
      setCameraState("starting");
      setFaceCount(0);

      // Reset calibration when starting a new camera session
      calibrationSamplesRef.current = [];
      baselineRef.current = null;
      calibrationStateRef.current = "not-calibrated";
      setCalibrationState("not-calibrated");

      lastLoggedMovementRef.current = null;
      lastLoggedFaceStatusRef.current = null;

      const stream =
        await navigator.mediaDevices.getUserMedia({
          video: {
            width: {
              ideal: 1280,
            },

            height: {
              ideal: 720,
            },

            facingMode: "user",
          },

          audio: false,
        });

      streamRef.current = stream;

      const video = videoRef.current;

      if (!video) {
        throw new Error(
          "Video element is not available."
        );
      }

      // Attach camera stream
      video.srcObject = stream;

      // Allow inline playback
      video.setAttribute(
        "playsinline",
        "true"
      );

      video.muted = true;

      await video.play();

      console.log(
        "Camera started:",
        video.videoWidth,
        "x",
        video.videoHeight
      );

      setCameraState("active");
      setMonitoringStatus("STARTED");

      // Initialize MediaPipe
      await startFaceDetection();

    } catch (error) {
      console.error(
        "Camera access error:",
        error
      );

      // Stop stream if startup failed
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });

        streamRef.current = null;
      }

      setCameraState("error");

      if (
        error.name === "NotAllowedError"
      ) {
        setErrorMessage(
          "Camera permission was denied. Please allow camera access and try again."
        );

      } else if (
        error.name === "NotFoundError"
      ) {
        setErrorMessage(
          "No camera was found on this device."
        );

      } else {
        setErrorMessage(
          "Unable to access the camera. Please check your camera settings."
        );
      }
    }
  };

  // --------------------------------------------------
  // CALIBRATE
  // --------------------------------------------------

  const startCalibration = () => {
    if (
      !isCameraActive ||
      detectorState !== "active" ||
      faceCount !== 1
    ) {
      return;
    }

    calibrationSamplesRef.current = [];

    baselineRef.current = null;

    calibrationStateRef.current = "calibrating";

    setCalibrationState("calibrating");
    setMonitoringStatus("ACTIVE");
    setErrorMessage("");

    console.log(
      "Calibration started. Please look straight at the screen."
    );
  };

  // --------------------------------------------------
// STOP MONITORING
// --------------------------------------------------

const stopMonitoring = () => {
  calibrationSamplesRef.current = [];
  baselineRef.current = null;

  calibrationStateRef.current = "not-calibrated";
  setCalibrationState("not-calibrated");

  setMonitoringStatus("START");
  setCurrentMovement("LOOKING STRAIGHT");

  movementStateRef.current = "NORMAL";
  movementCandidateRef.current = "NORMAL";
  movementFramesRef.current = 0;

  lastLoggedMovementRef.current = null;
  lastLoggedFaceStatusRef.current = null;

  console.log("Monitoring stopped.");
};


  // --------------------------------------------------
  // STOP CAMERA
  // --------------------------------------------------

  const stopCamera = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(
        animationFrameRef.current
      );

      animationFrameRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    calibrationSamplesRef.current = [];

    baselineRef.current = null;

    calibrationStateRef.current = "not-calibrated";

    setCalibrationState("not-calibrated");

    setCameraState("off");
    setMonitoringStatus("NOT STARTED");
    setDetectorState("idle");
    setFaceCount(0);
    setCalibrationState(
      "not-calibrated"
    );
    setErrorMessage("");
  };

  // --------------------------------------------------
  // CLEANUP
  // --------------------------------------------------

  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(
          animationFrameRef.current
        );
      }

      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });
      }
    };
  }, []);

  // --------------------------------------------------
  // UI STATE
  // --------------------------------------------------

  const isCameraActive =
    cameraState === "active";

  const isStarting =
    cameraState === "starting";

  // --------------------------------------------------
  // FACE STATUS
  // --------------------------------------------------

  const getFaceStatus = () => {
    if (detectorState === "loading") {
      return "INITIALIZING";
    }

    if (detectorState === "error") {
      return "ERROR";
    }

    if (detectorState !== "active") {
      return "WAITING";
    }

    if (faceCount === 0) {
      return "NO FACE";
    }

    if (faceCount === 1) {
      if (
        calibrationState === "calibrating"
      ) {
        return "CALIBRATING";
      }

      if (
        calibrationState === "calibrated"
      ) {
        return "CALIBRATED";
      }

      return "ONE FACE";
    }

    return "MULTIPLE FACES";
  };

  // --------------------------------------------------
  // CALIBRATION BUTTON TEXT
  // --------------------------------------------------

  const getCalibrationButtonText = () => {
    if (
      calibrationState === "calibrating"
    ) {
      const sampleCount =
        calibrationSamplesRef.current.length;

      return `Calibrating... ${sampleCount}/${CALIBRATION_SAMPLE_COUNT}`;
    }

    if (
      calibrationState === "calibrated"
    ) {
      return "Recalibrate";
    }

    return "Calibrate";
  };

  // --------------------------------------------------
  // RENDER
  // --------------------------------------------------

  return (
    <section className="camera-card">

      {/* Header */}

      <div className="card-header">

        <div>
          <h2>
            Live Camera
          </h2>

          <p>
            Real-time facial movement monitoring
          </p>
        </div>

        <span
          className={`camera-badge ${
            isCameraActive
              ? "camera-active"
              : ""
          }`}
        >
          {isStarting
            ? "STARTING..."
            : isCameraActive
              ? "CAMERA ACTIVE"
              : cameraState === "error"
                ? "CAMERA ERROR"
                : "CAMERA OFF"}
        </span>

      </div>


      {/* Camera View */}

      <div
        className={`camera-view ${
          isCameraActive
            ? "camera-view-active"
            : ""
        }`}
      >

        {/* Video stays mounted */}

        <video
          ref={videoRef}
          className="camera-video"
          autoPlay
          playsInline
          muted
          style={{
            display: isCameraActive
              ? "block"
              : "none",
          }}
        />

        {!isCameraActive && (
          <div className="camera-placeholder">

            <div className="camera-icon">
              ◉
            </div>

            <h3>
              {isStarting
                ? "Starting Camera..."
                : "Camera Not Started"}
            </h3>

            <p>
              {isStarting
                ? "Requesting camera access"
                : "Start the camera to begin monitoring"}
            </p>

          </div>
        )}

        {/* Camera overlay */}

        {isCameraActive && (
          <div className="camera-overlay">

            <span className="detection-status">

              {detectorState === "loading"
                ? "INITIALIZING..."
                : detectorState === "active"
                  ? "FACE DETECTION ACTIVE"
                  : "DETECTION ERROR"}

            </span>

            <span className="face-count">
              Faces: {faceCount}
            </span>

          </div>
        )}

      </div>


      {/* Error */}

      {errorMessage && (
        <div className="camera-error">
          {errorMessage}
        </div>
      )}


      {/* Face Status */}

      {isCameraActive && (
        <div className="face-status">

          <span>
            FACE STATUS
          </span>

          <strong>
            {getFaceStatus()}
          </strong>

        </div>
      )}


      {/* Controls */}

      <div className="controls">

        {!isCameraActive ? (

          <button
            className="btn btn-primary"
            onClick={startCamera}
            disabled={isStarting}
          >
            {isStarting
              ? "Starting..."
              : "Start Camera"}
          </button>

        ) : (

          <button
            className="btn btn-danger"
            onClick={stopCamera}
          >
            Stop Camera
          </button>

        )}


        <button
          className="btn"
          onClick={startCalibration}
          disabled={
            !isCameraActive ||
            detectorState !== "active" ||
            faceCount !== 1 ||
            calibrationState === "calibrating"
          }
        >
          {getCalibrationButtonText()}
        </button>

        <button
          className="btn btn-danger"
          onClick={stopMonitoring}
          disabled={
            calibrationState !== "calibrated"
          }
        >
          Stop Monitoring
        </button>

      </div>

    </section>
  );
}

export default CameraPanel;