import unittest
from dataclasses import dataclass

from pose_analysis import (
    ABSENT,
    SITTING,
    STANDING,
    UNKNOWN,
    PostureClassifier,
    classify_posture,
    estimate_knee_angle,
    estimate_thigh_angle,
)

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
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


def sitting_leg(hip_idx, knee_idx, ankle_idx):
    # Side profile of a seated person: thigh horizontal (knee forward of
    # the hip at the same height), shin dropping to the floor.
    return {
        hip_idx: Point(0.5, 0.5, 1.0),
        knee_idx: Point(0.65, 0.52, 1.0),
        ankle_idx: Point(0.63, 0.7, 1.0),
    }


def standing_leg(hip_idx, knee_idx, ankle_idx):
    # Hip, knee, and ankle in a vertical line: a straight standing leg.
    return {
        hip_idx: Point(0.5, 0.4, 1.0),
        knee_idx: Point(0.5, 0.6, 1.0),
        ankle_idx: Point(0.5, 0.8, 1.0),
    }


def visible_shoulders():
    return {
        LEFT_SHOULDER: Point(0.45, 0.2, 1.0),
        RIGHT_SHOULDER: Point(0.55, 0.2, 1.0),
    }


def visible_hips():
    return {
        LEFT_HIP: Point(0.45, 0.5, 1.0),
        RIGHT_HIP: Point(0.55, 0.5, 1.0),
    }


class KneeAngleTest(unittest.TestCase):
    def test_seated_knee_measures_bent(self):
        landmarks = make_landmarks(sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        angle = estimate_knee_angle(landmarks)
        self.assertLess(angle, 110.0)

    def test_straight_leg_measures_near_180_degrees(self):
        landmarks = make_landmarks(standing_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        angle = estimate_knee_angle(landmarks)
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_prefers_more_visible_leg(self):
        overrides = sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        overrides[LEFT_HIP].visibility = 0.2  # left leg barely visible
        overrides.update(standing_leg(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
        landmarks = make_landmarks(overrides)
        angle = estimate_knee_angle(landmarks)
        self.assertAlmostEqual(angle, 180.0, delta=1.0)

    def test_none_when_neither_leg_visible(self):
        overrides = sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        for point in overrides.values():
            point.visibility = 0.1
        landmarks = make_landmarks(overrides)
        self.assertIsNone(estimate_knee_angle(landmarks))


class ThighAngleTest(unittest.TestCase):
    def test_horizontal_thigh_measures_near_90(self):
        landmarks = make_landmarks(sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        self.assertGreater(estimate_thigh_angle(landmarks), 80.0)

    def test_vertical_thigh_measures_near_0(self):
        landmarks = make_landmarks(standing_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE))
        self.assertLess(estimate_thigh_angle(landmarks), 5.0)

    def test_works_without_ankle(self):
        # Ankle hidden behind a desk leg: knee angle is unavailable, but
        # the thigh direction still is.
        overrides = sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        overrides[LEFT_ANKLE].visibility = 0.1
        landmarks = make_landmarks(overrides)
        self.assertIsNone(estimate_knee_angle(landmarks))
        self.assertGreater(estimate_thigh_angle(landmarks), 80.0)

    def test_none_when_neither_thigh_visible(self):
        landmarks = make_landmarks(visible_hips())
        self.assertIsNone(estimate_thigh_angle(landmarks))


class ClassifyPostureTest(unittest.TestCase):
    def test_seated_side_profile_is_sitting(self):
        overrides = {
            **visible_shoulders(),
            **sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        }
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), SITTING)

    def test_straight_vertical_leg_is_standing(self):
        overrides = {
            **visible_shoulders(),
            **standing_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        }
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), STANDING)

    def test_sitting_with_ankle_hidden_still_sitting(self):
        # Standing-desk-adjacent clutter often hides the ankle; the thigh
        # alone must be enough to keep reading a seated person as seated.
        overrides = {
            **visible_shoulders(),
            **sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        }
        overrides[LEFT_ANKLE].visibility = 0.1
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), SITTING)

    def test_legs_hidden_is_unknown_not_sitting(self):
        # Standing behind the desk: torso visible, legs occluded. This
        # used to be "assume sitting", which read standing as sitting;
        # now it's UNKNOWN so the caller holds its previous reading.
        landmarks = make_landmarks({**visible_shoulders(), **visible_hips()})
        self.assertEqual(classify_posture(landmarks), UNKNOWN)

    def test_ambiguous_thigh_resolved_by_knee_angle(self):
        # Thigh ~40° off vertical (mid-stride territory): between the
        # thresholds, so the knee angle decides.
        overrides = {
            **visible_shoulders(),
            LEFT_HIP: Point(0.5, 0.4, 1.0),
            LEFT_KNEE: Point(0.5 + 0.13, 0.4 + 0.15, 1.0),  # ~41° from vertical
            LEFT_ANKLE: Point(0.5 + 0.26, 0.4 + 0.30, 1.0),  # straight leg
        }
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), STANDING)

    def test_ambiguous_thigh_without_ankle_is_unknown(self):
        overrides = {
            **visible_shoulders(),
            LEFT_HIP: Point(0.5, 0.4, 1.0),
            LEFT_KNEE: Point(0.63, 0.55, 1.0),  # ~41° from vertical
        }
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), UNKNOWN)

    def test_knee_above_hip_is_unknown_not_sitting(self):
        # Seen live: standing at the desk produced garbage legs with the
        # "knee" well above the hip (thigh ~140° from vertical). That's
        # implausible for sitting, so it must read as noise, not SITTING.
        overrides = {
            **visible_shoulders(),
            LEFT_HIP: Point(0.5, 0.5, 1.0),
            LEFT_KNEE: Point(0.6, 0.4, 1.0),  # above the hip
            LEFT_ANKLE: Point(0.7, 0.3, 1.0),
        }
        landmarks = make_landmarks(overrides)
        self.assertEqual(classify_posture(landmarks), UNKNOWN)

    def test_phantom_with_no_visible_landmarks_is_absent(self):
        landmarks = make_landmarks({})
        self.assertEqual(classify_posture(landmarks), ABSENT)

    def test_shoulder_only_phantom_on_empty_desk_is_absent(self):
        # An empty chair/desk can have something shoulder-height (a
        # headrest, a monitor, clothing) clear the visibility threshold
        # while no hips are ever detected.
        landmarks = make_landmarks(visible_shoulders())
        self.assertEqual(classify_posture(landmarks), ABSENT)

    def test_degenerate_torso_blob_is_absent(self):
        # Phantom detections on clutter collapse shoulders and hips into
        # one cluster; a real torso has meaningful shoulder-hip distance.
        landmarks = make_landmarks(
            {
                LEFT_SHOULDER: Point(0.5, 0.5, 1.0),
                RIGHT_SHOULDER: Point(0.51, 0.5, 1.0),
                LEFT_HIP: Point(0.5, 0.53, 1.0),
                RIGHT_HIP: Point(0.51, 0.53, 1.0),
            }
        )
        self.assertEqual(classify_posture(landmarks), ABSENT)

    def test_hips_above_shoulders_is_absent(self):
        # An upside-down "torso" isn't a person at a desk.
        landmarks = make_landmarks(
            {
                LEFT_SHOULDER: Point(0.45, 0.6, 1.0),
                RIGHT_SHOULDER: Point(0.55, 0.6, 1.0),
                LEFT_HIP: Point(0.45, 0.3, 1.0),
                RIGHT_HIP: Point(0.55, 0.3, 1.0),
            }
        )
        self.assertEqual(classify_posture(landmarks), ABSENT)


class PostureClassifierBaselineTest(unittest.TestCase):
    """The stateful classifier learns the sitting hip height from
    leg-confirmed frames and uses it to resolve UNKNOWN ones (flaky leg
    landmarks are the norm with a backlit desk scene)."""

    def _seated_frame(self):
        return make_landmarks(
            {**visible_shoulders(), **sitting_leg(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)}
        )

    def _legs_hidden_frame(self, hip_y):
        return make_landmarks(
            {
                LEFT_SHOULDER: Point(0.45, hip_y - 0.3, 1.0),
                RIGHT_SHOULDER: Point(0.55, hip_y - 0.3, 1.0),
                LEFT_HIP: Point(0.45, hip_y, 1.0),
                RIGHT_HIP: Point(0.55, hip_y, 1.0),
            }
        )

    def test_unknown_at_sitting_hip_height_resolves_to_sitting(self):
        c = PostureClassifier()
        self.assertEqual(c.classify(self._seated_frame()), SITTING)
        # Legs vanish but the hip stays at the learned height (0.5):
        self.assertEqual(c.classify(self._legs_hidden_frame(0.5)), SITTING)

    def test_unknown_with_hip_well_above_baseline_is_standing(self):
        c = PostureClassifier()
        c.classify(self._seated_frame())  # baseline: hip 0.5, torso ~0.3
        # Standing raises the hip by ~0.6-0.8 torso lengths:
        self.assertEqual(c.classify(self._legs_hidden_frame(0.28)), STANDING)

    def test_unknown_without_baseline_stays_unknown(self):
        c = PostureClassifier()
        self.assertEqual(c.classify(self._legs_hidden_frame(0.5)), UNKNOWN)

    def test_garbage_legs_at_sitting_height_resolve_to_sitting(self):
        # Seen live: knee hallucinated above the hip while actually
        # sitting. With a baseline, this must read sitting, not hold.
        c = PostureClassifier()
        c.classify(self._seated_frame())
        garbage = make_landmarks(
            {
                **visible_shoulders(),
                **visible_hips(),
                LEFT_KNEE: Point(0.6, 0.3, 1.0),  # "knee" at elbow height
            }
        )
        self.assertEqual(c.classify(garbage), SITTING)

    def test_baseline_not_trained_by_fallback_frames(self):
        # Only leg-confirmed sitting updates the baseline; frames the
        # baseline itself resolved must not drag it around.
        c = PostureClassifier()
        c.classify(self._seated_frame())
        baseline = c.sitting_hip_y
        c.classify(self._legs_hidden_frame(0.55))  # resolved by fallback
        self.assertEqual(c.sitting_hip_y, baseline)


if __name__ == "__main__":
    unittest.main()
