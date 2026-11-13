"""Video round-trip and failure checks without downloading or loading models."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import process_video as video


class ProcessVideoTests(unittest.TestCase):
    def test_round_trip_reuses_models_and_preserves_frames(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 4, (200, 100))
            self.assertTrue(writer.isOpened(), "MP4 encoder unavailable")
            try:
                for _ in range(3):
                    writer.write(np.full((100, 200, 3), 255, np.uint8))
            finally:
                writer.release()
            with patch.object(video, "ensure_model", return_value="model.pt"), \
                    patch.object(video, "YOLO") as yolo, \
                    patch.object(video, "create_reader") as reader:
                boxes = yolo.return_value.predict.return_value[0].boxes.data
                boxes.tolist.return_value = [[10, 40, 190, 80, 0.9, 0]]
                reader.return_value.recognize.return_value = [([], "MH12AB1234", 0.92)]
                output, count = video.process_video(source, root / "output")
                self.assertEqual(count, 3)
                yolo.assert_called_once()
                reader.assert_called_once()
                self.assertEqual(yolo.return_value.predict.call_count, 3)
                self.assertEqual(reader.return_value.recognize.call_count, 3)
            capture = cv2.VideoCapture(str(output))
            try:
                self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 4, places=2)
                decoded = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    self.assertEqual(frame.shape, (100, 200, 3))
                    self.assertLess(frame.min(), 100, "Annotations were not written")
                    decoded += 1
                self.assertEqual(decoded, 3)
            finally:
                capture.release()
            self.assertFalse(output.with_name("annotated.partial.mp4").exists())

    def test_invalid_video_does_not_load_models(self):
        with TemporaryDirectory() as folder, patch.object(video, "ensure_model") as model:
            source = Path(folder) / "missing.mp4"
            with self.assertRaisesRegex(ValueError, "not found"):
                video.process_video(source)
            source.write_text("not a video")
            with self.assertRaises(ValueError):
                video.process_video(source)
            model.assert_not_called()

    def test_bad_fps_and_early_end_are_errors(self):
        for fps, expected_error in [(0.0, "frame rate"), (4.0, "ended early")]:
            with self.subTest(fps=fps), TemporaryDirectory() as folder:
                source = Path(folder) / "input.mp4"
                source.touch()
                with patch.object(video.cv2, "VideoCapture") as capture_class, \
                        patch.object(video.cv2, "VideoWriter") as writer_class, \
                        patch.object(video, "ensure_model"), \
                        patch.object(video, "YOLO") as yolo, patch.object(video, "create_reader"):
                    capture = capture_class.return_value
                    capture.read.side_effect = [(True, np.zeros((100, 200, 3), np.uint8)), (False, None)]
                    capture.get.side_effect = [fps, 2.0]
                    yolo.return_value.predict.return_value[0].boxes.data.tolist.return_value = []
                    with self.assertRaisesRegex(ValueError, expected_error):
                        video.process_video(source, Path(folder) / "output")
                    capture.release.assert_called_once()
                    if fps > 0:
                        writer_class.return_value.release.assert_called_once()
                    self.assertEqual(list(Path(folder).rglob("annotated.mp4")), [])

    def test_encoder_failure_releases_resources(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "input.mp4"
            source.touch()
            with patch.object(video.cv2, "VideoCapture") as capture_class, \
                    patch.object(video.cv2, "VideoWriter") as writer_class, \
                    patch.object(video, "ensure_model"), \
                    patch.object(video, "YOLO"), patch.object(video, "create_reader"):
                capture = capture_class.return_value
                capture.read.return_value = (True, np.zeros((100, 200, 3), np.uint8))
                capture.get.side_effect = [4.0, 1.0]
                writer_class.return_value.isOpened.return_value = False
                with self.assertRaisesRegex(OSError, "Cannot create MP4"):
                    video.process_video(source, Path(folder) / "output")
                capture.release.assert_called_once()
                writer_class.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
