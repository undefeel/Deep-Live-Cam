import importlib
import unittest


def _load_model_downloader():
    return importlib.import_module("modules.model_downloader")


class ExternalModelUrlTests(unittest.TestCase):
    def test_hand_landmarker_resolves_to_google_storage(self):
        model_downloader = _load_model_downloader()
        self.assertEqual(
            model_downloader.resolve_url("hand_landmarker.task"),
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task",
        )

    def test_hf_models_still_resolve_to_hugging_face(self):
        model_downloader = _load_model_downloader()
        self.assertEqual(
            model_downloader.resolve_url("inswapper_128.onnx"),
            "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx",
        )

    def test_hand_landmarker_expected_size(self):
        model_downloader = _load_model_downloader()
        self.assertEqual(model_downloader.expected_size("hand_landmarker.task"), 7819105)


if __name__ == "__main__":
    unittest.main()
