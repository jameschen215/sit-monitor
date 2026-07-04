Here's how to do it yourself in Terminal:

Run in the foreground (see logs live, Ctrl+C to stop):
cd /Users/chenjian/repos/sit-monitor
PYTHONUNBUFFERED=1 ./venv/bin/python3 sit_monitor.py

Run in the background, logging to a file:
cd /Users/chenjian/repos/sit-monitor
PYTHONUNBUFFERED=1 nohup ./venv/bin/python3 sit_monitor.py > sit_monitor.log 2>&1 &
That prints a PID (or find it later with pgrep -f sit_monitor.py).

Tail the logs while it runs in the background:
tail -f /Users/chenjian/repos/sit-monitor/sit_monitor.log
(Ctrl+C just exits the tail, doesn't stop the service.)

Stop it:
pkill -f sit_monitor.py

Note: PYTHONUNBUFFERED=1 matters — without it, Python buffers stdout when it's not a terminal (e.g. writing to a file or nohup), so log lines lag behind in chunks instead of appearing as they happen.

The instance I started earlier (PID 95632) is still running and writing to /private/tmp/claude-501/-Users-chenjian-repos-sit-monitor/73067dec-e587-4f13-90c6-c162507a46b1/tasks/btffuanza.output — if you want a clean handoff, kill that one first (kill 95632) before starting your own so you don't have two instances fighting over the camera connection.