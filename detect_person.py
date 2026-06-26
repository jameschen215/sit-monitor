import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

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

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

print('Running pose detection... Press "q" to quit.')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: cannot read frame")
        break

    # MediaPipe works with RGB, OpenCV uses BGR by default
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if results.pose_landmarks:
        status = "Person detected"
        color = (0, 255, 0)
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
