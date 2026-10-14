"""Hand portal live effect.

Two hands in the frame define a quadrilateral through their thumb tips
(MediaPipe landmark 4) and index tips (landmark 8). Inside that quadrilateral
a filter from FILTERS is applied to the frame; outside it the frame is left
untouched, and a thin white outline is drawn along its edges. The quadrilateral
may self-intersect (a "bow-tie"); it is filled with the even-odd-free fillPoly
rule of cv2.fillPoly, so a crossed quad fills the way OpenCV paints it.

A pinch of one hand (all five fingertips gathered together) switches the
active filter to the next one in FILTERS. The pinch is only detected while the
webcam preview is running, and it is rate-limited by PinchCycler.

MediaPipe is imported lazily, so this module imports and works (as a no-op
effect with no hands) on machines without mediapipe installed.
"""

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import modules.globals
from modules import model_downloader

Point = Tuple[float, float]
Hand = List[Point]  # 21 MediaPipe landmarks in pixel coordinates
FilterFn = Callable[[np.ndarray], np.ndarray]

HAND_MODEL_NAME = "hand_landmarker.task"
WRIST = 0
MIDDLE_MCP = 9
THUMB_TIP = 4
INDEX_TIP = 8
FINGERTIPS = (4, 8, 12, 16, 20)
PINCH_RATIO = 0.25
OUTLINE_COLOR = (255, 255, 255)
OUTLINE_THICKNESS = 2

# --- Filters -----------------------------------------------------------------
# Each filter takes a BGR uint8 image and returns a BGR uint8 image of the same
# shape. New filters are added with register_filter() below the base set; the
# insertion order of FILTERS is the cycling order.

FILTERS: Dict[str, FilterFn] = {}


def register_filter(name: str, fn: FilterFn) -> None:
    """Add (or replace) a filter under ``name``."""
    FILTERS[name] = fn


def filter_names() -> List[str]:
    """Names of all registered filters, in cycling order."""
    return list(FILTERS)


def next_filter(name: str) -> str:
    """Return the filter after ``name``, wrapping around; unknown -> first."""
    names = filter_names()
    if not names:
        return name
    if name not in names:
        return names[0]
    return names[(names.index(name) + 1) % len(names)]


def _purple(frame: np.ndarray) -> np.ndarray:
    tint = np.full_like(frame, (180, 0, 160))
    return cv2.addWeighted(frame, 0.4, tint, 0.6, 0)


def _grayscale(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _negative(frame: np.ndarray) -> np.ndarray:
    return 255 - frame


register_filter("purple", _purple)
register_filter("grayscale", _grayscale)
register_filter("negative", _negative)

# --- Geometry ----------------------------------------------------------------


def portal_corners(hands: Sequence[Hand]) -> Optional[np.ndarray]:
    """Quadrilateral [A[8], B[8], B[4], A[4]] for the first two hands.

    Hands are ordered by wrist x (left to right): A is the left one. Returns
    None when fewer than two hands are given.
    """
    if len(hands) < 2:
        return None
    left, right = sorted(hands[:2], key=lambda hand: hand[WRIST][0])
    corners = [left[INDEX_TIP], right[INDEX_TIP], right[THUMB_TIP], left[THUMB_TIP]]
    return np.rint(np.array(corners, dtype=np.float64)).astype(np.int32)


def apply_filter_in_polygon(
    frame: np.ndarray, polygon: np.ndarray, filter_name: str
) -> np.ndarray:
    """Return a copy of ``frame`` with ``filter_name`` applied inside ``polygon``.

    Only the polygon's bounding box (clipped to the frame) is filtered. A white
    outline is drawn on the polygon edges. Degenerate polygons, polygons fully
    outside the frame and unknown filter names return an unchanged copy. The
    input frame is never modified.
    """
    out = frame.copy()
    fn = FILTERS.get(filter_name)
    if fn is None:
        return out

    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    # cv2.boundingRect reports 1 px for a single point, so measure the extent
    # of the points directly to catch zero-area polygons.
    extent = points.max(axis=0) - points.min(axis=0)
    if extent[0] == 0 or extent[1] == 0:
        return out

    box_x, box_y, box_w, box_h = cv2.boundingRect(points)
    frame_h, frame_w = frame.shape[:2]
    x0, y0 = max(0, box_x), max(0, box_y)
    x1, y1 = min(frame_w, box_x + box_w), min(frame_h, box_y + box_h)
    if x1 <= x0 or y1 <= y0:
        return out

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local = points - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(mask, [local], 255)

    filtered = fn(out[y0:y1, x0:x1].copy())
    inside = mask > 0
    region = out[y0:y1, x0:x1]
    region[inside] = filtered[inside]

    cv2.polylines(
        out,
        [points.reshape(-1, 1, 2)],
        True,
        OUTLINE_COLOR,
        OUTLINE_THICKNESS,
        cv2.LINE_AA,
    )
    return out


# --- Pinch -------------------------------------------------------------------


def _distance(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def is_pinch(hand: Hand, ratio: float = PINCH_RATIO) -> bool:
    """True when all five fingertips are within ``ratio * hand_size`` of their centroid.

    hand_size is the distance between the wrist (0) and the middle finger
    base (9). A hand with zero size is never considered pinched.
    """
    hand_size = _distance(hand[WRIST], hand[MIDDLE_MCP])
    if hand_size == 0:
        return False
    tips = [hand[i] for i in FINGERTIPS]
    centroid = (
        sum(tip[0] for tip in tips) / len(tips),
        sum(tip[1] for tip in tips) / len(tips),
    )
    limit = ratio * hand_size
    return all(_distance(tip, centroid) < limit for tip in tips)


class PinchCycler:
    """Reports the rising edge of a pinch, at most once per ``cooldown_s``."""

    def __init__(
        self,
        cooldown_s: float = 0.8,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._was_pinched = False
        self._last_fire: Optional[float] = None

    def update(self, hands: Sequence[Hand]) -> bool:
        """Feed the hands seen in one frame; True when a new pinch should trigger."""
        pinched = any(is_pinch(hand) for hand in hands)
        rising = pinched and not self._was_pinched
        self._was_pinched = pinched
        if not rising:
            return False
        now = self._clock()
        if self._last_fire is not None and now - self._last_fire < self._cooldown_s:
            return False
        self._last_fire = now
        return True


_PINCH_CYCLER = PinchCycler()

# --- Detection ---------------------------------------------------------------

_LOCAL = threading.local()  # per-thread (mediapipe_module, landmarker) pair
_STATE_LOCK = threading.Lock()
_UNAVAILABLE = False  # set once detection cannot work in this process
_WARNED = False


def _warn_once(message: str) -> None:
    global _WARNED
    with _STATE_LOCK:
        if _WARNED:
            return
        _WARNED = True
    print(f"[DLC.HAND_PORTAL] {message}")


def _mark_unavailable(message: str) -> None:
    global _UNAVAILABLE
    _UNAVAILABLE = True
    _warn_once(message)


def ensure_model() -> Optional[str]:
    """Path to hand_landmarker.task, downloading it if needed."""
    return model_downloader.ensure_model(HAND_MODEL_NAME)


def _thread_landmarker():
    """Return (mediapipe module, HandLandmarker) for this thread, or None."""
    if _UNAVAILABLE:
        return None
    cached = getattr(_LOCAL, "pair", None)
    if cached is not None:
        return cached
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as error:
        _mark_unavailable(f"mediapipe is not available, hand portal disabled: {error}")
        return None

    model_path = ensure_model()
    if model_path is None:
        _mark_unavailable("hand_landmarker.task is missing, hand portal disabled")
        return None

    try:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        landmarker = vision.HandLandmarker.create_from_options(options)
    except Exception as error:
        _mark_unavailable(f"could not create HandLandmarker, hand portal disabled: {error}")
        return None

    _LOCAL.pair = (mp, landmarker)
    return _LOCAL.pair


def detect_hands(frame: np.ndarray) -> List[Hand]:
    """Detect up to two hands in a BGR frame; landmarks in pixel coordinates.

    Never raises: returns [] when mediapipe or the model is unavailable or
    detection fails.
    """
    pair = _thread_landmarker()
    if pair is None:
        return []
    mp, landmarker = pair

    frame_h, frame_w = frame.shape[:2]
    try:
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)
    except Exception as error:
        _warn_once(f"hand detection failed for a frame: {error}")
        return []

    return [
        [(lm.x * frame_w, lm.y * frame_h) for lm in hand]
        for hand in result.hand_landmarks
    ]


# --- Entry point -------------------------------------------------------------


def apply_hand_portal(
    frame: np.ndarray, hands: Optional[Sequence[Hand]] = None
) -> np.ndarray:
    """Apply the hand portal effect to ``frame``.

    ``hands`` defaults to detect_hands(frame). While the webcam preview runs, a
    pinch advances modules.globals.hand_portal_filter to the next filter.
    Returns ``frame`` itself when there is no portal to draw.
    """
    if hands is None:
        hands = detect_hands(frame)

    if modules.globals.webcam_preview_running and _PINCH_CYCLER.update(hands):
        modules.globals.hand_portal_filter = next_filter(
            modules.globals.hand_portal_filter
        )

    corners = portal_corners(hands)
    if corners is None:
        return frame
    return apply_filter_in_polygon(frame, corners, modules.globals.hand_portal_filter)
