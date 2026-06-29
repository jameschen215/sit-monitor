import os
import time
import subprocess
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -- Configuration --
CHECK_INTERVAL = 5  # seconds between each detection check
SITTING_LIMIT = 2700  # 45 minutes in seconds
AWAY_THRESHOLD = 300  # 5 minutes away resets the timer
MODEL_PATH = "pose_landmarker.task"

# -- Setup MediaPipe --
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# -- Set up Camera --
# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("rtsp://admin:TUHXOF@192.168.0.109:554/h264/ch1/main/av_stream")
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# -- State --
sitting_seconds = 0
away_seconds = 0

print('Sit monitor started. Press "q" to quit.')

last_check = 0

while True:
    now = time.time()

    # Only grab a frame every CHECK_INTERVAL seconds
    if now - last_check < CHECK_INTERVAL:
        time.sleep(0.5)
        continue

    last_check = now

    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame")
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    person_present = bool(results.pose_landmarks)

    if person_present:
        sitting_seconds += CHECK_INTERVAL
        away_seconds = 0
    else:
        away_seconds += CHECK_INTERVAL
        if away_seconds >= AWAY_THRESHOLD:
            sitting_seconds = 0  # reset if away long enough

    # Display status
    minutes = sitting_seconds // 60
    status = f"Sitting: {minutes}m {sitting_seconds % 60}s"
    color = (0, 255, 0) if person_present else (0, 0, 255)
    # cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # cv.imshow("Sit Monitor", frame)

    print(
        f"Person present: {person_present} | Sitting: {sitting_seconds // 60}m {sitting_seconds % 60}s"
    )

    # Trigger alert
    if sitting_seconds >= SITTING_LIMIT:
        subprocess.run(
            [
                "notify-send",
                "Sitting Alert",
                "You've been sitting for 45 minutes. Time to stand up!",
            ]
        )
        subprocess.run(["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
        sitting_seconds = 0  # reset after alert

    # if cv.waitKey(1) & 0xFF == ord("q"):
    #     break

    # time.sleep(CHECK_INTERVAL)

cap.release()
# cv.destroyAllWindows()
