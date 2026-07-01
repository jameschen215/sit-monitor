import unittest

from sit_state import SitMonitor, WORKING, WARNING, RESTING


def make_monitor(**overrides):
    defaults = dict(check_interval=5, sitting_limit=2700, rest_threshold=300, confirm_checks=1)
    defaults.update(overrides)
    return SitMonitor(**defaults)


def feed(monitor, readings):
    fired = []
    for i, sitting in enumerate(readings):
        alerts = monitor.tick(sitting)
        if alerts:
            fired.append((i, alerts))
    return fired


class WorkingPhasePauseTest(unittest.TestCase):
    def test_sitting_ticks_up_while_under_limit(self):
        m = make_monitor()
        feed(m, [True] * 20)  # 100s
        self.assertEqual(m.sitting_seconds, 100)
        self.assertEqual(m.phase, WORKING)

    def test_standing_before_limit_pauses_without_alerts_or_rest_tracking(self):
        m = make_monitor()
        feed(m, [True] * 20)  # 100s sitting
        fired = feed(m, [False] * 30)  # 150s standing, well under the 45min limit
        self.assertEqual(fired, [])
        self.assertEqual(m.sitting_seconds, 100)  # untouched
        self.assertEqual(m.standing_seconds, 0)  # rest timer untouched entirely
        self.assertEqual(m.phase, WORKING)

    def test_resumes_sitting_timer_from_paused_value(self):
        m = make_monitor()
        feed(m, [True] * 20)  # 100s
        feed(m, [False] * 10)  # paused
        feed(m, [True] * 4)  # resumes: +20s
        self.assertEqual(m.sitting_seconds, 120)


class WarningPhaseTest(unittest.TestCase):
    def test_crossing_limit_fires_nag_and_keeps_ticking(self):
        m = make_monitor()
        fired = feed(m, [True] * (2760 // 5))  # 46 minutes continuously
        self.assertEqual(
            [a for _, a in fired],
            [["你已经坐了45分钟了，需要休息一下！"], ["你已经坐了46分钟了，需要休息一下！"]],
        )
        self.assertEqual(m.phase, WARNING)


class RestingPhaseTest(unittest.TestCase):
    def _over_limit_monitor(self):
        m = make_monitor()
        feed(m, [True] * (2700 // 5))  # exactly at the limit
        return m

    def test_standing_after_limit_enters_resting_silently(self):
        m = self._over_limit_monitor()
        fired = feed(m, [False] * (240 // 5))  # 4 minutes standing, under rest_threshold
        self.assertEqual(fired, [])
        self.assertEqual(m.phase, RESTING)
        self.assertEqual(m.standing_seconds, 240)
        self.assertEqual(m.sitting_seconds, 2700)  # frozen while standing

    def test_you_can_sit_now_fires_once_at_threshold(self):
        m = self._over_limit_monitor()
        fired = feed(m, [False] * (600 // 5))  # 10 minutes standing, well past 5 min
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0][1], ["你可以坐下了！"])

    def test_sitting_back_down_after_enough_rest_resets_everything(self):
        m = self._over_limit_monitor()
        feed(m, [False] * (420 // 5))  # 7 minutes standing
        fired = feed(m, [True])
        self.assertEqual(fired, [(0, ["很好！你已经休息了7分钟了。"])])
        self.assertEqual(m.sitting_seconds, 0)
        self.assertEqual(m.standing_seconds, 0)
        self.assertEqual(m.phase, WORKING)

    def test_sitting_back_down_before_enough_rest_returns_to_warning(self):
        m = self._over_limit_monitor()
        feed(m, [False] * (120 // 5))  # 2 minutes standing, short of 5 min
        fired = feed(m, [True] * 12)  # sits back down for a minute
        self.assertEqual(m.phase, WARNING)
        self.assertEqual(m.standing_seconds, 120)  # rest credit preserved, not reset
        self.assertTrue(any("需要休息" in a for _, alerts in fired for a in alerts))

    def test_rest_credit_accumulates_across_interruptions(self):
        m = self._over_limit_monitor()
        feed(m, [False] * (120 // 5))  # 2 min standing
        feed(m, [True] * 6)  # interrupt: sit for 30s (back to warning)
        fired = feed(m, [False] * (180 // 5))  # stand again: +3 min -> total 5 min
        self.assertEqual(fired, [(35, ["你可以坐下了！"])])
        fired = feed(m, [True])  # now sit back down: enough total rest -> reset
        self.assertEqual(fired, [(0, ["很好！你已经休息了5分钟了。"])])
        self.assertEqual(m.sitting_seconds, 0)
        self.assertEqual(m.standing_seconds, 0)


class DebounceTest(unittest.TestCase):
    def test_single_flicker_does_not_flip_posture(self):
        m = make_monitor(confirm_checks=2)
        feed(m, [True] * 10)
        feed(m, [False])  # single miss, not yet confirmed
        self.assertEqual(m.sitting_seconds, 50 + 5)  # still counted as sitting
        feed(m, [True])  # recovers before confirming
        self.assertEqual(m.phase, WORKING)

    def test_two_consecutive_flips_confirm_standing(self):
        m = make_monitor(confirm_checks=2)
        feed(m, [True] * 10)
        feed(m, [False, False])
        self.assertEqual(m.effective_sitting, False)


if __name__ == "__main__":
    unittest.main()
