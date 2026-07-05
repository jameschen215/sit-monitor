import sys
import time
import queue
import threading
import subprocess
from pathlib import Path

import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera_config import get_rtsp_url
from pose_analysis import (
    ABSENT,
    SITTING,
    UNKNOWN,
    classify_posture,
    estimate_knee_angle,
    estimate_thigh_angle,
)
from sit_state import SitMonitor

# -- Configuration --
CHECK_INTERVAL = 5  # seconds between each detection check
SITTING_LIMIT = 2700  # 45 minutes in seconds
REST_THRESHOLD = 300  # 5 minutes of accumulated standing satisfies a rest
MODEL_PATH = "pose_landmarker.task"
RTSP_URL = get_rtsp_url()
EDGE_TTS_PATH = Path(__file__).resolve().parent / "venv" / "bin" / "edge-tts"
ALERT_MP3 = Path("/tmp/sit_alert.mp3")

# -- Setup MediaPipe --
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    # Above the 0.5 defaults: at 0.5 MediaPipe hallucinated poses on the
    # empty room (shoulder/hip visibility cleared threshold too), which
    # read as someone sitting there for hours.
    min_pose_detection_confidence=0.65,
    min_pose_presence_confidence=0.65,
)
detector = vision.PoseLandmarker.create_from_options(options)

# -- Set up Camera --
cap = cv.VideoCapture(RTSP_URL)


def send_alert(text):
    # Remove the previous alert's audio first: if TTS fails (offline,
    # Azure hiccup) we must not replay a stale file with the wrong message.
    ALERT_MP3.unlink(missing_ok=True)
    tts = subprocess.run(
        [
            str(EDGE_TTS_PATH),
            "--voice",
            "zh-CN-shaanxi-XiaoniNeural",
            "--text",
            text,
            "--write-media",
            str(ALERT_MP3),
        ]
    )
    tts_ok = tts.returncode == 0 and ALERT_MP3.exists()
    if not tts_ok:
        print(f"Alert TTS failed (exit {tts.returncode}), skipping audio: {text}")
    if sys.platform == "darwin":
        escaped_text = text.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{escaped_text}" with title "久坐提醒"',
            ]
        )
        if tts_ok:
            subprocess.run(["afplay", str(ALERT_MP3)])
    else:
        subprocess.run(["notify-send", "久坐提醒", text])
        if tts_ok:
            subprocess.run(["mpg123", "-a", "pulse", str(ALERT_MP3)])


# Alerts run on their own thread so TTS synthesis and audio playback
# (5-10s each) never stall the main loop — cap.grab() must keep draining
# the RTSP buffer or frames go stale and the stream reconnects.
alert_queue = queue.Queue()


def _alert_worker():
    while True:
        text = alert_queue.get()
        try:
            send_alert(text)
        except Exception as exc:
            # A broken alert path (missing binary, audio failure) must not
            # kill the worker; alerting would silently stop forever.
            print(f"Alert failed: {exc}")


threading.Thread(target=_alert_worker, daemon=True).start()


# -- State --
monitor = SitMonitor(
    check_interval=CHECK_INTERVAL,
    sitting_limit=SITTING_LIMIT,
    rest_threshold=REST_THRESHOLD,
)

print("Sit monitor started.")

last_check = 0
last_tick_time = None

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

    thigh_angle = knee_angle = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks[0]
        posture = classify_posture(landmarks)
        thigh_angle = estimate_thigh_angle(landmarks)
        knee_angle = estimate_knee_angle(landmarks)
    else:
        posture = ABSENT

    if posture == UNKNOWN:
        # Person present but legs hidden (standing behind the desk, half
        # out of frame): hold the current posture rather than guess.
        person_sitting = monitor.effective_sitting
    else:
        person_sitting = posture == SITTING

    # Credit real wall-clock time to the state machine so slow ticks
    # (detection latency, reconnect sleeps) don't undercount; tick()
    # caps runaway gaps and treats suspend-length ones as a fresh start.
    if last_tick_time is None:
        elapsed = CHECK_INTERVAL
    else:
        elapsed = int(round(now - last_tick_time))
    last_tick_time = now

    for alert_text in monitor.tick(person_sitting, elapsed):
        alert_queue.put(alert_text)

    def fmt_angle(angle):
        return f"{angle:.0f}°" if angle is not None else "-"

    print(
        f"Phase: {monitor.phase} | Posture: {posture} | "
        f"Thigh: {fmt_angle(thigh_angle)} | Knee: {fmt_angle(knee_angle)} | "
        f"Sitting: {monitor.sitting_seconds // 60}m {monitor.sitting_seconds % 60}s | "
        f"Standing: {monitor.standing_seconds // 60}m {monitor.standing_seconds % 60}s"
    )
