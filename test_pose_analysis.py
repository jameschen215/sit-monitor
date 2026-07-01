import unittest
from dataclasses import dataclass

from pose_analysis import estimate_knee_angle, is_sitting

LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28


@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


def make_landmarks(overrides):
    """33 landmarks, all at origin with visibility 0 by default; overrides
    is a dict of {index: Point} for the ones that matter to a test."""
    landmarks = [Point(0.0, 0.0, 0.0) for _ in range(33)]
    for idx, point in overrides.items():
        landmarks[idx] = point
    return landmarks


def bent_leg(hip_idx, knee_idx, ankle_idx):
    # Hip directly above knee, ankle out to the side: a sharp ~90 degree bend.
    return {
        hip_idx: Point(0.5, 0.4, 1.0),
        knee_idx: Point(0.5, 0.6, 1.0),
        ankle_idx: Point(0.3, 0.6, 1.0),
    }


def straight_leg(hip_idx, knee_idx, ankle_idx):
    # Hip, knee, and ankle in a vertical line: a straight standing leg.
    return {
        hip_idx: Point(0.5, 0.4, 1.0),
        knee_idx: Point(0.5, 0.6, 1.0),
        ankle_idx: Point(0.5, 0.8, 1.0),
    }


class KneeAngleTest(unittest.TestCase):
    def test_bent_knee_measures_near_90_degrees(self):
        landmarks = make_landmarks(bent_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        angle = estimate_knee_angle(landmarks)
        self.assertAlmostEqual(angle, 90.0, delta=1.0)

    def test_straight_leg_measures_near_180_degrees(self):
        landmarks = make_landmarks(straight_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        angle = estimate_knee_angle(landmarks)
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_prefers_more_visible_leg(self):
        overrides = bent_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        overrides[LEFT_HIP].visibility = 0.2  # left leg barely visible
        overrides.update(straight_leg(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
        landmarks = make_landmarks(overrides)
        angle = estimate_knee_angle(landmarks)
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_none_when_neither_leg_visible(self):
        overrides = bent_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        for point in overrides.values():
            point.visibility = 0.1
        landmarks = make_landmarks(overrides)
        self.assertIsNone(estimate_knee_angle(landmarks))


class IsSittingTest(unittest.TestCase):
    def test_bent_knee_is_sitting(self):
        landmarks = make_landmarks(bent_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        self.assertTrue(is_sitting(landmarks))

    def test_straight_leg_is_standing(self):
        landmarks = make_landmarks(straight_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        self.assertFalse(is_sitting(landmarks))

    def test_falls_back_to_true_when_legs_not_visible(self):
        # e.g. a desk-mounted camera that only sees the upper body.
        landmarks = make_landmarks({})
        self.assertTrue(is_sitting(landmarks))


if __name__ == "__main__":
    unittest.main()
