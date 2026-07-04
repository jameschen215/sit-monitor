from dataclasses import dataclass

WORKING = "working"
WARNING = "warning"
RESTING = "resting"


@dataclass
class SitMonitor:
    check_interval: int = 5
    sitting_limit: int = 2700  # 45 minutes; past this, standing up counts as rest
    rest_threshold: int = 300  # 5 minutes of accumulated standing satisfies a rest
    confirm_checks: int = 2  # consecutive disagreeing reads before flipping posture

    sitting_seconds: int = 0
    standing_seconds: int = 0
    can_sit_now_notified: bool = False
    last_warning_minute: int = -1

    effective_sitting: bool = True
    posture_streak: int = 0

    def _debounce(self, person_sitting: bool) -> bool:
        # A single flickered detection (bad frame, brief occlusion) shouldn't
        # flip posture; require confirm_checks consecutive disagreeing reads.
        if person_sitting == self.effective_sitting:
            self.posture_streak = 0
        else:
            self.posture_streak += 1
            if self.posture_streak >= self.confirm_checks:
                self.effective_sitting = person_sitting
                self.posture_streak = 0
        return self.effective_sitting

    @property
    def phase(self) -> str:
        if self.sitting_seconds < self.sitting_limit:
            return WORKING
        return WARNING if self.effective_sitting else RESTING

    def tick(self, person_sitting: bool, elapsed: int | None = None) -> list[str]:
        """Advance the state machine given a raw sitting/standing reading.
        elapsed is the real wall-clock seconds since the previous tick, so
        slow ticks (alert playback, RTSP reconnects) don't undercount time;
        it defaults to check_interval for callers that tick on a fixed
        cadence. Returns any alert messages to announce."""
        if elapsed is None:
            elapsed = self.check_interval
        sitting = self._debounce(person_sitting)
        alerts = []

        if sitting:
            if self.standing_seconds >= self.rest_threshold:
                # Enough rest was accumulated (possibly across separate
                # standing stints) before they sat back down: fresh start.
                # This applies under the limit too — a lunch break at
                # minute 30 shouldn't leave the timer at 30 minutes.
                alerts.append("欢迎大师兄归位！")
                self.sitting_seconds = 0
                self.standing_seconds = 0
                self.can_sit_now_notified = False
                self.last_warning_minute = -1
            else:
                self.sitting_seconds += elapsed
                if self.sitting_seconds >= self.sitting_limit:
                    minute = self.sitting_seconds // 60
                    if minute > self.last_warning_minute:
                        self.last_warning_minute = minute
                        alerts.append(f"你已经坐了{minute}分钟了，需要休息一下！")
        else:
            self.standing_seconds += elapsed
            # Only announce "rest complete" during an over-limit rest cycle;
            # someone who left the room before any warning shouldn't have
            # this spoken to an empty chair.
            if (
                self.sitting_seconds >= self.sitting_limit
                and self.standing_seconds >= self.rest_threshold
                and not self.can_sit_now_notified
            ):
                minutes_rested = self.standing_seconds // 60
                alerts.append(f"你已经休息了{minutes_rested}分钟了，可以工作啦！")
                self.can_sit_now_notified = True

        return alerts
