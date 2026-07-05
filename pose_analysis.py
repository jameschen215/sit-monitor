import math

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, LEFT_KNEE, LEFT_ANKLE = 23, 25, 27
RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE = 24, 26, 28

VISIBILITY_THRESHOLD = 0.5
SITTING_KNEE_ANGLE_MAX = 140.0  # degrees: bent knee = sitting, straight leg = standing
SITTING_THIGH_ANGLE_MIN = 50.0  # thigh this far off vertical = sitting
SITTING_THIGH_ANGLE_MAX = 120.0  # past this the knee is above the hip —
# implausible for sitting, so treat the reading as noise
STANDING_THIGH_ANGLE_MAX = 35.0  # thigh this close to vertical = standing
MIN_TORSO_DROP = 0.02  # shoulders must sit above hips by this much (normalized)
MIN_TORSO_LENGTH = 0.08  # smaller than this is landmark noise, not a torso

# Classification results. UNKNOWN means a person is present but their legs
# are too hidden to judge posture — the caller should hold its previous
# reading rather than guess, because guessing "sitting" reads someone
# standing behind the desk as sitting, and guessing "standing" hands out
# rest credit while they sit.
SITTING = "sitting"
STANDING = "standing"
UNKNOWN = "unknown"
ABSENT = "absent"


def _visible_mean(landmarks, indices, visibility_threshold):
    """Mean (x, y) of the confidently-visible landmarks among indices,
    or None if none qualify."""
    points = [
        landmarks[i] for i in indices if landmarks[i].visibility >= visibility_threshold
    ]
    if not points:
        return None
    return (
        sum(p.x for p in points) / len(points),
        sum(p.y for p in points) / len(points),
    )


def _person_present(landmarks, visibility_threshold=VISIBILITY_THRESHOLD) -> bool:
    """Require a confidently-visible shoulder AND hip, arranged like an
    actual torso, before trusting that MediaPipe detected a real person.

    Shoulder alone isn't enough: an empty chair/desk can have something
    shoulder-height (headrest, monitor, clothing) trip the threshold, and
    a phantom blob's "hip" can too. Beyond visibility, a real torso has
    shoulders above hips with meaningful distance between them; phantom
    detections on background clutter tend to collapse into a degenerate
    cluster, so cheap geometry rules them out without breaking the
    legitimate "legs hidden under the desk" case."""
    shoulder = _visible_mean(
        landmarks, (LEFT_SHOULDER, RIGHT_SHOULDER), visibility_threshold
    )
    hip = _visible_mean(landmarks, (LEFT_HIP, RIGHT_HIP), visibility_threshold)
    if shoulder is None or hip is None:
        return False
    # y grows downward: hips below shoulders means a positive drop.
    if hip[1] - shoulder[1] < MIN_TORSO_DROP:
        return False
    return math.hypot(hip[0] - shoulder[0], hip[1] - shoulder[1]) >= MIN_TORSO_LENGTH


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


def _pick_side(landmarks, left_indices, right_indices, visibility_threshold):
    """Return the landmark tuple for whichever side's chain is more
    visible, or None if neither side clears the threshold throughout."""
    left_vis = min(landmarks[i].visibility for i in left_indices)
    right_vis = min(landmarks[i].visibility for i in right_indices)
    if left_vis < visibility_threshold and right_vis < visibility_threshold:
        return None
    indices = left_indices if left_vis >= right_vis else right_indices
    return tuple(landmarks[i] for i in indices)


def estimate_knee_angle(landmarks, visibility_threshold=VISIBILITY_THRESHOLD):
    """Return the hip-knee-ankle angle for whichever leg is more visible,
    or None if neither full leg is visible enough to trust (e.g. ankle
    hidden under a desk)."""
    leg = _pick_side(
        landmarks,
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        visibility_threshold,
    )
    if leg is None:
        return None
    return _angle(*leg)


def estimate_thigh_angle(landmarks, visibility_threshold=VISIBILITY_THRESHOLD):
    """Angle of the hip->knee vector from vertical, in degrees, for
    whichever thigh is more visible; None if neither is visible enough.

    0 = thigh pointing straight down (standing), ~90 = thigh horizontal
    (sitting). Needs only hip and knee, so it still works when the ankle
    is hidden behind a desk leg or chair base — the situation where the
    full knee angle is unavailable."""
    thigh = _pick_side(
        landmarks,
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE),
        visibility_threshold,
    )
    if thigh is None:
        return None
    hip, knee = thigh
    dx = knee.x - hip.x
    dy = knee.y - hip.y  # y grows downward; positive = knee below hip
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), dy))


def classify_posture(
    landmarks, visibility_threshold=VISIBILITY_THRESHOLD
) -> str:
    """Classify one person's pose landmarks as SITTING, STANDING,
    UNKNOWN, or ABSENT.

    Thigh orientation is the primary signal: sitting puts the thigh near
    horizontal and standing puts it near vertical, and it stays
    measurable with just hip + knee. The knee angle breaks ties in the
    ambiguous band between the two thresholds (e.g. mid-stride while
    walking). With no leg information at all the posture is UNKNOWN, not
    assumed — see the constant docs above."""
    if not _person_present(landmarks, visibility_threshold):
        return ABSENT
    thigh_angle = estimate_thigh_angle(landmarks, visibility_threshold)
    if thigh_angle is None:
        return UNKNOWN
    if thigh_angle > SITTING_THIGH_ANGLE_MAX:
        return UNKNOWN
    if thigh_angle >= SITTING_THIGH_ANGLE_MIN:
        return SITTING
    if thigh_angle <= STANDING_THIGH_ANGLE_MAX:
        return STANDING
    knee_angle = estimate_knee_angle(landmarks, visibility_threshold)
    if knee_angle is None:
        return UNKNOWN
    return SITTING if knee_angle <= SITTING_KNEE_ANGLE_MAX else STANDING
