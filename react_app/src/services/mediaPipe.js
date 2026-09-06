import {
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

let faceLandmarker = null;

export async function initializeFaceLandmarker() {
  if (faceLandmarker) {
    return faceLandmarker;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(
  "/mediapipe"
);

    faceLandmarker = await FaceLandmarker.createFromOptions(
      vision,
      {
        baseOptions: {
          modelAssetPath: "/models/face_landmarker.task",
          delegate: "CPU",
        },

        runningMode: "VIDEO",

        numFaces: 2,

        minFaceDetectionConfidence: 0.5,

        minFacePresenceConfidence: 0.5,

        minTrackingConfidence: 0.5,

        outputFaceBlendshapes: false,

        outputFacialTransformationMatrixes: true,
      }
    );

    console.log("MediaPipe Face Landmarker initialized successfully.");

    return faceLandmarker;

  } catch (error) {
    console.error(
      "MediaPipe initialization failed:",
      error
    );

    faceLandmarker = null;

    throw error;
  }
}

export function detectFaceLandmarks(
  videoElement,
  timestamp
) {
  if (!faceLandmarker) {
    return null;
  }

  return faceLandmarker.detectForVideo(
    videoElement,
    timestamp
  );
}

export function getFaceLandmarker() {
  return faceLandmarker;
}