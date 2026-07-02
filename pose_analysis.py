import math

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

VISIBILITY_THRESHOLD = 0.5
SITTING_KNEE_ANGLE_MAX = 140.0  # degrees: bent knee = sitting, straight leg = standing


def _person_present(landmarks, visibility_threshold=VISIBILITY_THRESHOLD) -> bool:
    """Require a confidently-visible shoulder AND hip before trusting that
    MediaPipe detected a real person, rather than a low-confidence phantom
    detection (background clutter, furniture) that happens to clear the
    pose detector's threshold.

    Shoulder alone isn't enough: an empty chair/desk can have something
    shoulder-height (headrest, monitor, clothing) trip the threshold, and
    since it has no legs either, is_sitting's "legs not visible -> assume
    sitting" fallback would then read it as a person sitting there forever.
    A real person's hip is anatomically tied to their shoulder in a way a
    phantom blob won't coincidentally match, so requiring both is a cheap
    way to rule out the empty-desk case without breaking the legitimate
    "legs hidden under the desk" case, where hips are still visible."""
    shoulder_visible = (
        landmarks[LEFT_SHOULDER].visibility >= visibility_threshold
        or landmarks[RIGHT_SHOULDER].visibility >= visibility_threshold
    )
    hip_visible = (
        landmarks[LEFT_HIP].visibility >= visibility_threshold
        or landmarks[RIGHT_HIP].visibility >= visibility_threshold
    )
    return shoulder_visible and hip_visible


def _angle(a, b, c) -> float:
    """Angle ABC in degrees, with b as the vertex."""
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)
    if mag_ab == 0 or mag_cb == 0:
        return 180.0
    cos_angle = (ab[0] * cb[0] + ab[1] * cb[1]) / (mag_ab * mag_cb)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def estimate_knee_angle(landmarks, visibility_threshold=VISIBILITY_THRESHOLD):
    """Return the hip-knee-ankle angle for whichever leg is more visible,
    or None if neither leg is visible enough to trust (e.g. hidden under
    a desk)."""

    def leg_visibility(hip_idx, knee_idx, ankle_idx):
        return min(
            landmarks[hip_idx].visibility,
            landmarks[knee_idx].visibility,
            landmarks[ankle_idx].visibility,
        )

    left_vis = leg_visibility(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    right_vis = leg_visibility(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)

    if left_vis < visibility_threshold and right_vis < visibility_threshold:
        return None

    if left_vis >= right_vis:
        return _angle(landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
    return _angle(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])


def is_sitting(
    landmarks,
    angle_threshold=SITTING_KNEE_ANGLE_MAX,
    visibility_threshold=VISIBILITY_THRESHOLD,
) -> bool:
    """Classify sitting vs standing from one person's pose landmarks.

    Returns False outright if there's no confidently-visible person at
    all (see _person_present) — a low-confidence phantom detection
    shouldn't default to "sitting". Only once a real person is confirmed
    present does it fall back to True (assume sitting) when neither leg
    is visible enough to measure a knee angle, since a bent-knee signal
    isn't available in every camera framing (e.g. a desk-mounted camera
    that only sees the upper body)."""
    if not _person_present(landmarks, visibility_threshold):
        return False
    angle = estimate_knee_angle(landmarks, visibility_threshold)
    if angle is None:
        return True
    return angle <= angle_threshold
