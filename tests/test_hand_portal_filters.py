import unittest

import numpy as np

from modules import hand_portal


class HandPortalFiltersTests(unittest.TestCase):
    def test_filter_names_order(self):
        names = hand_portal.filter_names()
        expected = [
            "purple",
            "grayscale",
            "negative",
            "sketch",
            "thermal",
            "sepia",
            "pixelate",
            "cool",
        ]
        self.assertEqual(names[:8], expected)

    def test_filters_output_shape_dtype_and_no_mutation(self):
        shapes = [(1, 1, 3), (7, 5, 3), (120, 160, 3)]
        rng = np.random.default_rng(42)

        for name in hand_portal.filter_names():
            fn = hand_portal.FILTERS[name]
            for shape in shapes:
                frame = rng.integers(0, 256, shape, dtype=np.uint8)
                original = frame.copy()

                out = fn(frame)

                self.assertIsInstance(out, np.ndarray, f"{name} returned non-ndarray")
                self.assertEqual(out.dtype, np.uint8, f"{name} dtype mismatch")
                self.assertEqual(out.shape, shape, f"{name} shape mismatch for {shape}")
                np.testing.assert_array_equal(
                    frame,
                    original,
                    err_msg=f"{name} mutated input frame for shape {shape}",
                )
