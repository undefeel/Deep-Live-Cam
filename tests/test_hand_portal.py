import contextlib
import importlib
import io
import sys
import threading
import unittest
from unittest import mock

import numpy as np


def _load_hand_portal():
    return importlib.import_module("modules.hand_portal")


hand_portal = _load_hand_portal()

OPEN_TIPS = {4: (-60, -60), 8: (-30, -110), 12: (0, -120), 16: (30, -110), 20: (60, -80)}


def make_hand(wrist_x, wrist_y=300, tips=None):
    """21 synthetic landmarks; hand_size (wrist -> landmark 9) is 100 px."""
    hand = [(float(wrist_x), float(wrist_y))] * 21
    hand[9] = (float(wrist_x), float(wrist_y - 100))
    offsets = OPEN_TIPS if tips is None else tips
    for index, (dx, dy) in offsets.items():
        hand[index] = (float(wrist_x + dx), float(wrist_y + dy))
    return hand


def make_pinch_hand(wrist_x, wrist_y=300):
    """All five fingertips gathered within a few px of one point."""
    gathered = {
        index: (0 + jitter, -90 + jitter)
        for index, jitter in zip((4, 8, 12, 16, 20), (-3, 2, 0, 3, -1))
    }
    return make_hand(wrist_x, wrist_y, tips=gathered)


class PortalCornersTests(unittest.TestCase):
    def test_corner_order_uses_wrist_x_not_input_order(self):
        left = make_hand(100)
        right = make_hand(400)
        corners = hand_portal.portal_corners([right, left])
        expected = np.array(
            [left[8], right[8], right[4], left[4]], dtype=np.int32
        )
        self.assertEqual(corners.dtype, np.int32)
        self.assertEqual(corners.shape, (4, 2))
        np.testing.assert_array_equal(corners, expected)

    def test_none_for_fewer_than_two_hands(self):
        self.assertIsNone(hand_portal.portal_corners([]))
        self.assertIsNone(hand_portal.portal_corners([make_hand(100)]))

    def test_extra_hands_are_ignored(self):
        first, second, third = make_hand(100), make_hand(400), make_hand(250)
        corners = hand_portal.portal_corners([first, second, third])
        np.testing.assert_array_equal(
            corners,
            np.array([first[8], second[8], second[4], first[4]], dtype=np.int32),
        )


class FilterRegistryTests(unittest.TestCase):
    def test_base_filters_registered_in_order(self):
        names = hand_portal.filter_names()
        self.assertEqual(names[:3], ["purple", "grayscale", "negative"])

    def test_next_filter_wraps_around(self):
        names = hand_portal.filter_names()
        self.assertEqual(hand_portal.next_filter(names[-1]), names[0])
        self.assertEqual(hand_portal.next_filter("purple"), "grayscale")
        self.assertEqual(hand_portal.next_filter("negative"), "purple")

    def test_next_filter_unknown_name_gives_first(self):
        self.assertEqual(hand_portal.next_filter("nope"), hand_portal.filter_names()[0])

    def test_base_filters_keep_shape_and_dtype(self):
        frame = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        for name in ("purple", "grayscale", "negative"):
            out = hand_portal.FILTERS[name](frame)
            self.assertEqual(out.shape, frame.shape, name)
            self.assertEqual(out.dtype, np.uint8, name)


class ApplyFilterInPolygonTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((200, 200, 3), 100, dtype=np.uint8)
        self.square = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)

    def test_inside_changes_outside_unchanged(self):
        out = hand_portal.apply_filter_in_polygon(self.frame, self.square, "negative")
        self.assertEqual(out[100, 100].tolist(), [155, 155, 155])
        self.assertEqual(out[20, 20].tolist(), [100, 100, 100])
        self.assertEqual(out[180, 180].tolist(), [100, 100, 100])
        np.testing.assert_array_equal(out[:40, :], self.frame[:40, :])

    def test_outline_is_drawn_white(self):
        out = hand_portal.apply_filter_in_polygon(self.frame, self.square, "negative")
        self.assertGreaterEqual(int(out[100, 48:53].max()), 200)

    def test_input_frame_not_mutated(self):
        original = self.frame.copy()
        hand_portal.apply_filter_in_polygon(self.frame, self.square, "negative")
        np.testing.assert_array_equal(self.frame, original)

    def test_self_intersecting_bow_tie_is_filled(self):
        bow_tie = np.array([[50, 50], [150, 150], [150, 50], [50, 150]], dtype=np.int32)
        out = hand_portal.apply_filter_in_polygon(self.frame, bow_tie, "negative")
        self.assertFalse(np.array_equal(out, self.frame))
        self.assertEqual(out[100, 140].tolist(), [155, 155, 155])

    def test_degenerate_polygon_returns_unchanged_copy(self):
        point = np.array([[80, 80], [80, 80], [80, 80]], dtype=np.int32)
        out = hand_portal.apply_filter_in_polygon(self.frame, point, "negative")
        np.testing.assert_array_equal(out, self.frame)
        self.assertIsNot(out, self.frame)

    def test_polygon_fully_outside_returns_unchanged(self):
        outside = np.array([[300, 300], [350, 300], [350, 350], [300, 350]], dtype=np.int32)
        out = hand_portal.apply_filter_in_polygon(self.frame, outside, "negative")
        np.testing.assert_array_equal(out, self.frame)

    def test_unknown_filter_returns_unchanged(self):
        out = hand_portal.apply_filter_in_polygon(self.frame, self.square, "no-such-filter")
        np.testing.assert_array_equal(out, self.frame)
        self.assertIsNot(out, self.frame)

    def test_partially_outside_polygon_is_clipped(self):
        partly = np.array([[150, 50], [260, 50], [260, 150], [150, 150]], dtype=np.int32)
        out = hand_portal.apply_filter_in_polygon(self.frame, partly, "negative")
        self.assertEqual(out[100, 190].tolist(), [155, 155, 155])
        self.assertEqual(out[100, 20].tolist(), [100, 100, 100])


class IsPinchTests(unittest.TestCase):
    def test_pinched_hand_is_pinch(self):
        self.assertTrue(hand_portal.is_pinch(make_pinch_hand(100)))

    def test_open_hand_is_not_pinch(self):
        self.assertFalse(hand_portal.is_pinch(make_hand(100)))

    def test_zero_hand_size_is_not_pinch(self):
        hand = [(5.0, 5.0)] * 21
        self.assertFalse(hand_portal.is_pinch(hand))


class PinchCyclerTests(unittest.TestCase):
    def test_fires_once_per_hold_and_respects_cooldown(self):
        now = [0.0]
        cycler = hand_portal.PinchCycler(cooldown_s=0.8, clock=lambda: now[0])
        pinch = [make_pinch_hand(100)]
        open_hand = [make_hand(100)]

        now[0] = 0.0
        self.assertTrue(cycler.update(pinch))
        now[0] = 0.1
        self.assertFalse(cycler.update(pinch))  # still held: no new edge
        now[0] = 0.2
        self.assertFalse(cycler.update(open_hand))
        now[0] = 0.3
        self.assertFalse(cycler.update(pinch))  # rising edge, but inside cooldown
        now[0] = 0.4
        self.assertFalse(cycler.update(open_hand))
        now[0] = 0.8
        self.assertTrue(cycler.update(pinch))  # cooldown over since the first fire at 0.0

    def test_no_hands_never_fires(self):
        cycler = hand_portal.PinchCycler(clock=lambda: 0.0)
        self.assertFalse(cycler.update([]))


class DetectHandsTests(unittest.TestCase):
    def test_returns_empty_when_mediapipe_import_fails(self):
        # Force "import mediapipe" to fail and reset the module's cached state so
        # the test does not depend on what other tests did earlier in this process.
        blocked = {"mediapipe": None}
        stdout = io.StringIO()
        with mock.patch.dict(sys.modules, blocked), \
                mock.patch.object(hand_portal, "_LOCAL", threading.local()), \
                mock.patch.object(hand_portal, "_UNAVAILABLE", False), \
                mock.patch.object(hand_portal, "_WARNED", False), \
                contextlib.redirect_stdout(stdout):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            self.assertEqual(hand_portal.detect_hands(frame), [])
            self.assertEqual(hand_portal.detect_hands(frame), [])
        warnings = [line for line in stdout.getvalue().splitlines() if "[DLC.HAND_PORTAL]" in line]
        self.assertEqual(len(warnings), 1)


class ApplyHandPortalTests(unittest.TestCase):
    def setUp(self):
        import modules.globals as globals_

        self.globals_ = globals_
        self.saved = (
            globals_.webcam_preview_running,
            globals_.hand_portal_filter,
        )
        globals_.webcam_preview_running = False
        globals_.hand_portal_filter = "purple"
        self.frame = np.full((200, 200, 3), 100, dtype=np.uint8)

    def tearDown(self):
        (
            self.globals_.webcam_preview_running,
            self.globals_.hand_portal_filter,
        ) = self.saved

    def test_one_hand_returns_frame_unchanged(self):
        out = hand_portal.apply_hand_portal(self.frame, [make_hand(100)])
        np.testing.assert_array_equal(out, self.frame)

    def test_two_hands_apply_selected_filter(self):
        left = make_hand(60, 150)
        right = make_hand(140, 150)
        out = hand_portal.apply_hand_portal(self.frame, [left, right])
        self.assertFalse(np.array_equal(out, self.frame))
        np.testing.assert_array_equal(self.frame, np.full((200, 200, 3), 100, dtype=np.uint8))

    def test_pinch_cycles_filter_only_while_preview_runs(self):
        left = make_hand(60, 150)
        right = make_hand(140, 150)
        pinch_left = make_pinch_hand(60, 150)
        cycler = hand_portal.PinchCycler(clock=lambda: 0.0)

        with mock.patch.object(hand_portal, "_PINCH_CYCLER", cycler):
            self.globals_.webcam_preview_running = False
            hand_portal.apply_hand_portal(self.frame, [pinch_left, right])
            self.assertEqual(self.globals_.hand_portal_filter, "purple")

            self.globals_.webcam_preview_running = True
            hand_portal.apply_hand_portal(self.frame, [pinch_left, right])
            self.assertEqual(self.globals_.hand_portal_filter, "grayscale")
            hand_portal.apply_hand_portal(self.frame, [left, right])
            self.assertEqual(self.globals_.hand_portal_filter, "grayscale")


if __name__ == "__main__":
    unittest.main()
