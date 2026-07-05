# sit-monitor

Watches an RTSP camera with MediaPipe pose detection and reminds you to
stand up after 45 minutes of sitting, using spoken alerts (edge-tts) and
desktop notifications. Runs on Linux (notify-send + mpg123) and macOS
(osascript + afplay).

## How it works

- `sit_monitor.py` — main loop: grabs a frame every 5 seconds, classifies
  sitting/standing, feeds the result to the state machine, and plays any
  alerts on a background thread.
- `pose_analysis.py` — classifies pose landmarks as sitting / standing /
  unknown / absent. Thigh orientation (hip→knee vs vertical) is the
  primary signal, knee angle breaks ties, and phantom-detection guards
  require a visible shoulder AND hip arranged like a real torso.
- `sit_state.py` — the timer state machine: 45 min sitting limit,
  5 min accumulated standing counts as a rest and resets the timer; a
  suspend-length gap between ticks resets it quietly.
- `camera_config.py` — reads `RTSP_URL` from `.env`.

The camera should see your full body in side profile so thigh and knee
geometry are measurable; when a person is present but their legs are
hidden the posture reading is held rather than guessed.

## Setup

```sh
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cp .env.example .env   # then fill in RTSP_URL
```

`pose_landmarker.task` (the MediaPipe model) is committed in the repo.

## Run

Foreground:

```sh
PYTHONUNBUFFERED=1 ./venv/bin/python3 sit_monitor.py
```

As a systemd service (Linux):

```sh
sudo cp sit-monitor.service /etc/systemd/system/
sudo systemctl enable --now sit-monitor.service
journalctl -u sit-monitor.service -f   # follow logs
```

## Tests

```sh
./venv/bin/python3 -m unittest test_sit_state test_pose_analysis
```

## Diagnostics

- `test_cam.py` — view the raw RTSP stream to check the camera connection.
- `detect_person.py` — live pose-detection overlay to check framing and
  landmark visibility.
