import unittest

from sit_state import SitMonitor, STATE_AWAY, STATE_SITTING


def make_monitor(**overrides):
    defaults = dict(check_interval=5, sitting_limit=2700, away_threshold=300)
    defaults.update(overrides)
    return SitMonitor(**defaults)


def feed(monitor, readings):
    """Feed a sequence of raw presence readings, return list of (tick_index, alerts)."""
    fired = []
    for i, present in enumerate(readings):
        alerts = monitor.tick(present)
        if alerts:
            fired.append((i, alerts))
    return fired


class SittingOvertimeTest(unittest.TestCase):
    def test_alert_fires_every_minute_past_limit(self):
        m = make_monitor()
        fired = feed(m, [True] * (2760 // 5))  # 46 minutes continuously present
        self.assertEqual([a for _, a in fired], [["你已经坐了45分钟了，请站起来活动一下！"], ["你已经坐了46分钟了，请站起来活动一下！"]])
        self.assertEqual(m.state, STATE_SITTING)
        self.assertEqual(m.sitting_seconds, 2760)


class LeavingResetsSitTimerTest(unittest.TestCase):
    def test_leaving_immediately_resets_and_starts_away(self):
        m = make_monitor(confirm_checks=1)
        feed(m, [True] * 20)  # 100s sitting
        self.assertEqual(m.sitting_seconds, 100)
        feed(m, [False])
        self.assertEqual(m.state, STATE_AWAY)
        self.assertEqual(m.sitting_seconds, 0)
        self.assertEqual(m.away_seconds, 0)


class AwayNagTest(unittest.TestCase):
    def test_nag_fires_once_per_minute_while_present_during_break(self):
        m = make_monitor(confirm_checks=1)
        m.state = STATE_AWAY
        readings = [True] * (240 // 5)  # 4 minutes, present throughout the break
        fired = feed(m, readings)
        self.assertEqual(
            [a[0] for _, a in fired],
            [
                "你还需要休息4分钟，请继续站立活动！",
                "你还需要休息3分钟，请继续站立活动！",
                "你还需要休息2分钟，请继续站立活动！",
                "你还需要休息1分钟，请继续站立活动！",
            ],
        )

    def test_silent_while_properly_absent_during_break(self):
        m = make_monitor(confirm_checks=1)
        m.state = STATE_AWAY
        m.effective_present = False
        fired = feed(m, [False] * (240 // 5))
        self.assertEqual(fired, [])

    def test_you_can_sit_now_fires_once_then_stays_silent(self):
        m = make_monitor(confirm_checks=1)
        m.state = STATE_AWAY
        m.effective_present = False
        fired = feed(m, [False] * (600 // 5))  # 10 minutes away, well past 5 min
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][1], ["你可以坐下了！"])

    def test_returning_after_break_reports_actual_rest_duration(self):
        m = make_monitor(confirm_checks=1)
        m.state = STATE_AWAY
        m.effective_present = False
        feed(m, [False] * (420 // 5))  # 7 minutes away
        fired = feed(m, [True])
        self.assertEqual(fired, [(0, ["很好！你已经休息了7分钟了。"])])
        self.assertEqual(m.state, STATE_SITTING)
        self.assertEqual(m.sitting_seconds, 0)
        self.assertEqual(m.away_seconds, 0)

    def test_returning_exactly_at_threshold_skips_you_can_sit_now(self):
        m = make_monitor(confirm_checks=1)
        m.state = STATE_AWAY
        m.effective_present = False
        feed(m, [False] * (295 // 5))
        fired = feed(m, [True])  # crosses 300s while present
        self.assertEqual(fired, [(0, ["很好！你已经休息了5分钟了。"])])


class DebounceTest(unittest.TestCase):
    def test_single_flicker_does_not_flip_state(self):
        m = make_monitor(confirm_checks=2)
        feed(m, [True] * 10)
        feed(m, [False])  # single miss, should not yet register
        self.assertEqual(m.state, STATE_SITTING)
        self.assertGreater(m.sitting_seconds, 0)
        feed(m, [True])  # recovers before confirming
        self.assertEqual(m.state, STATE_SITTING)

    def test_two_consecutive_misses_confirm_absence(self):
        m = make_monitor(confirm_checks=2)
        feed(m, [True] * 10)
        feed(m, [False, False])
        self.assertEqual(m.state, STATE_AWAY)


class MinuteTrackingSurvivesFullCycleTest(unittest.TestCase):
    def test_no_stale_minute_state_across_multiple_sit_away_cycles(self):
        m = make_monitor(confirm_checks=1)
        # First sitting session goes past the limit, generating alerts.
        feed(m, [True] * (2760 // 5))
        self.assertEqual(m.last_sit_alert_minute, 46)

        # Leave, take the mandatory break, and come back.
        feed(m, [False] * (600 // 5))  # 10 minutes away (past the 5 min minimum)
        fired = feed(m, [True])  # returns, starts a fresh sitting session
        self.assertEqual(m.state, STATE_SITTING)
        self.assertTrue(fired and "休息" in fired[0][1][0])

        # New sitting session should alert again at its own 45-minute mark,
        # not be blocked by the stale last_sit_alert_minute from before.
        fired = feed(m, [True] * (2700 // 5))
        self.assertTrue(any("45分钟" in a for _, alerts in fired for a in alerts))


if __name__ == "__main__":
    unittest.main()
