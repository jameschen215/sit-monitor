from dataclasses import dataclass

STATE_SITTING = "sitting"
STATE_AWAY = "away"


@dataclass
class SitMonitor:
    check_interval: int = 5
    sitting_limit: int = 2700
    away_threshold: int = 300
    confirm_checks: int = 2  # consecutive disagreeing reads before flipping presence

    state: str = STATE_SITTING
    sitting_seconds: int = 0
    away_seconds: int = 0
    break_complete_notified: bool = False
    last_sit_alert_minute: int = -1
    last_away_nag_minute: int = -1

    effective_present: bool = True
    presence_streak: int = 0

    def _debounce(self, person_present: bool) -> bool:
        # A single flickered detection (bad frame, brief occlusion) shouldn't
        # flip state; require confirm_checks consecutive disagreeing reads.
        if person_present == self.effective_present:
            self.presence_streak = 0
        else:
            self.presence_streak += 1
            if self.presence_streak >= self.confirm_checks:
                self.effective_present = person_present
                self.presence_streak = 0
        return self.effective_present

    def tick(self, person_present: bool) -> list[str]:
        """Advance the state machine by one check_interval given a raw
        presence reading. Returns any alert messages to announce."""
        present = self._debounce(person_present)
        alerts = []

        if self.state == STATE_SITTING:
            if present:
                self.sitting_seconds += self.check_interval
                minute = self.sitting_seconds // 60
                if (
                    self.sitting_seconds >= self.sitting_limit
                    and minute > self.last_sit_alert_minute
                ):
                    self.last_sit_alert_minute = minute
                    alerts.append(f"你已经坐了{minute}分钟了，请站起来活动一下！")
            else:
                # Left the seat: sit timer resets immediately and a
                # mandatory break starts, however long they'd been sitting.
                self.sitting_seconds = 0
                self.state = STATE_AWAY
                self.away_seconds = 0
                self.break_complete_notified = False
                self.last_away_nag_minute = -1
        else:  # STATE_AWAY
            self.away_seconds += self.check_interval
            minute = self.away_seconds // 60

            if self.away_seconds < self.away_threshold:
                if present and minute >= 1 and minute > self.last_away_nag_minute:
                    self.last_away_nag_minute = minute
                    remaining = self.away_threshold // 60 - minute
                    alerts.append(f"你还需要休息{remaining}分钟，请继续站立活动！")
            else:
                if present:
                    alerts.append(f"很好！你已经休息了{minute}分钟了。")
                    self.state = STATE_SITTING
                    self.sitting_seconds = 0
                    self.away_seconds = 0
                    self.last_sit_alert_minute = -1
                elif not self.break_complete_notified:
                    alerts.append("你可以坐下了！")
                    self.break_complete_notified = True

        return alerts
