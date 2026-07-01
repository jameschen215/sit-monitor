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

    def tick(self, person_sitting: bool) -> list[str]:
        """Advance the state machine by one check_interval given a raw
        sitting/standing reading. Returns any alert messages to announce."""
        sitting = self._debounce(person_sitting)
        over_limit = self.sitting_seconds >= self.sitting_limit
        alerts = []

        if not over_limit:
            if sitting:
                self.sitting_seconds += self.check_interval
            # Standing while still under the limit: just pause. No rest
            # tracking, no alerts, resumes from this value when they sit
            # back down.
            return alerts

        if sitting:
            if self.standing_seconds >= self.rest_threshold:
                # Enough rest was accumulated (possibly across separate
                # standing stints) before they sat back down: fresh start.
                minutes_rested = self.standing_seconds // 60
                alerts.append(f"很好！你已经休息了{minutes_rested}分钟了。")
                self.sitting_seconds = 0
                self.standing_seconds = 0
                self.can_sit_now_notified = False
                self.last_warning_minute = -1
            else:
                self.sitting_seconds += self.check_interval
                minute = self.sitting_seconds // 60
                if minute > self.last_warning_minute:
                    self.last_warning_minute = minute
                    alerts.append(f"你已经坐了{minute}分钟了，需要休息一下！")
        else:
            self.standing_seconds += self.check_interval
            if self.standing_seconds >= self.rest_threshold and not self.can_sit_now_notified:
                alerts.append("你可以坐下了！")
                self.can_sit_now_notified = True

        return alerts
