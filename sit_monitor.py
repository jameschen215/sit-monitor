import time
import subprocess
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_config import get_rtsp_url
from pose_analysis import is_sitting
from sit_state import SitMonitor

# -- Configuration --
CHECK_INTERVAL = 5  # seconds between each detection check
SITTING_LIMIT = 2700  # 45 minutes in seconds
REST_THRESHOLD = 300  # 5 minutes of accumulated standing satisfies a rest
MODEL_PATH = "pose_landmarker.task"
RTSP_URL = get_rtsp_url()

# -- Setup MediaPipe --
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options, output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# -- Set up Camera --
cap = cv.VideoCapture(RTSP_URL)


def send_alert(text):
    subprocess.run(
        [
            "/home/james/repos/sit-monitor/venv/bin/edge-tts",
            "--voice",
            "zh-CN-shaanxi-XiaoniNeural",
            "--text",
            text,
            "--write-media",
            "/tmp/sit_alert.mp3",
        ]
    )
    subprocess.run(["notify-send", "久坐提醒", text])
    subprocess.run(["mpg123", "-a", "pulse", "/tmp/sit_alert.mp3"])


# -- State --
monitor = SitMonitor(
    check_interval=CHECK_INTERVAL,
    sitting_limit=SITTING_LIMIT,
    rest_threshold=REST_THRESHOLD,
)

print('Sit monitor started. Press "q" to quit.')

last_check = 0

while True:
    now = time.time()

    # Keep draining the RTSP decoder's internal buffer so it doesn't build
    # up a backlog while we're not reading; otherwise the next retrieve()
    # after a multi-second gap returns a stale, buffered frame instead of
    # the live one.
    grabbed = cap.grab()
    if not grabbed:
        print("Error: cannot read frame, reconnecting...")
        cap.release()
        time.sleep(5)
        cap = cv.VideoCapture(RTSP_URL)
        last_check = 0
        continue

    # Only decode a frame every CHECK_INTERVAL seconds
    if now - last_check < CHECK_INTERVAL:
        time.sleep(0.1)
        continue

    last_check = now

    ret, frame = cap.retrieve()
    if not ret:
        print("Error: cannot read frame, reconnecting...")
        cap.release()
        time.sleep(5)
        cap = cv.VideoCapture(RTSP_URL)
        last_check = 0
        continue

    frame = cv.resize(frame, (640, 360))  # 16:9 比例
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    person_sitting = bool(results.pose_landmarks) and is_sitting(results.pose_landmarks[0])

    for alert_text in monitor.tick(person_sitting):
        send_alert(alert_text)

    print(
        f"Phase: {monitor.phase} | Pose detected: {bool(results.pose_landmarks)} | "
        f"Sitting posture: {person_sitting} | "
        f"Sitting: {monitor.sitting_seconds // 60}m {monitor.sitting_seconds % 60}s | "
        f"Standing: {monitor.standing_seconds // 60}m {monitor.standing_seconds % 60}s"
    )


cap.release()
