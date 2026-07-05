import os
import cv2 as cv
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_config import get_rtsp_url
from pose_analysis import ABSENT, classify_posture, estimate_knee_angle, estimate_thigh_angle

# Download the pose landmarker model if not present
MODEL_PATH = "pose_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        MODEL_PATH,
    )
    print("Model downloaded.")

# Set up the pose landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv.VideoCapture(get_rtsp_url())

print('Running pose detection... Press "q" to quit.')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: cannot read frame")
        break

    frame = cv.resize(frame, (640, 360))  # 16:9 比例
    # MediaPipe works with RGB, OpenCV uses BGR by default
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        posture = classify_posture(landmarks)
        thigh = estimate_thigh_angle(landmarks)
        knee = estimate_knee_angle(landmarks)
        status = (
            f"{posture} | thigh: {thigh:.0f}" if thigh is not None else f"{posture} | thigh: -"
        )
        status += f" | knee: {knee:.0f}" if knee is not None else " | knee: -"
        color = (0, 0, 255) if posture == ABSENT else (0, 255, 0)
    else:
        status = "No person detected"
        color = (0, 0, 255)

    # Display status on frame
    cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv.imshow("Pose Detection", frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
