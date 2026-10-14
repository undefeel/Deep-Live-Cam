import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from modules import imread_unicode, imwrite_unicode
from modules.processors.frame import hand_portal as processor
from modules.processors.frame.core import (
    FRAME_PROCESSORS_INTERFACE,
    load_frame_processor_module,
)


def _frame():
    return np.full((8, 8, 3), 40, dtype=np.uint8)


class HandPortalProcessorTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, True)

    def test_loads_through_frame_processor_loader(self):
        module = load_frame_processor_module("hand_portal")

        self.assertEqual(module.NAME, "DLC.HAND-PORTAL")
        for name in FRAME_PROCESSORS_INTERFACE:
            self.assertTrue(hasattr(module, name), name)
        self.assertTrue(hasattr(module, "process_frame_v2"))

    def test_process_frame_delegates_to_apply_hand_portal(self):
        frame = _frame()
        expected = np.zeros_like(frame)

        with mock.patch.object(
            processor.hand_portal, "apply_hand_portal", return_value=expected
        ) as apply:
            result = processor.process_frame(None, frame, target_face=None)

        apply.assert_called_once_with(frame)
        self.assertIs(result, expected)

    def test_process_frame_v2_delegates_to_apply_hand_portal(self):
        frame = _frame()
        expected = np.ones_like(frame)

        with mock.patch.object(
            processor.hand_portal, "apply_hand_portal", return_value=expected
        ) as apply:
            result = processor.process_frame_v2(frame)

        apply.assert_called_once_with(frame)
        self.assertIs(result, expected)

    def test_process_image_reads_processes_and_writes(self):
        target_path = os.path.join(self.tmp_dir, "in.png")
        output_path = os.path.join(self.tmp_dir, "out.png")
        self.assertTrue(imwrite_unicode(target_path, _frame()))

        def invert(frame, *_args, **_kwargs):
            return 255 - frame

        with mock.patch.object(
            processor.hand_portal, "apply_hand_portal", side_effect=invert
        ):
            processor.process_image(None, target_path, output_path)

        self.assertTrue(os.path.isfile(output_path))
        written = imread_unicode(output_path)
        self.assertIsNotNone(written)
        np.testing.assert_array_equal(written, 255 - _frame())

    def test_pre_check_fails_without_model(self):
        with mock.patch.object(
            processor.hand_portal, "ensure_model", return_value=None
        ), mock.patch.object(processor, "update_status") as status:
            self.assertFalse(processor.pre_check())

        status.assert_called_once_with(
            "Hand landmarker model is unavailable.", "DLC.HAND-PORTAL"
        )

    def test_pre_check_passes_with_model(self):
        with mock.patch.object(
            processor.hand_portal, "ensure_model", return_value="/models/hand.task"
        ):
            self.assertTrue(processor.pre_check())

    def test_pre_start_is_always_true(self):
        self.assertTrue(processor.pre_start())


if __name__ == "__main__":
    unittest.main()
