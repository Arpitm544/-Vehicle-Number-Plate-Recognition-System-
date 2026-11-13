"""Detect and read plates in a video, then save an annotated MP4."""

import argparse
import math
from pathlib import Path
import tempfile

import cv2

from detect_plates import YOLO, annotate_plates, ensure_model
from read_plate import create_reader


def process_video(video_path, output_root=Path("outputs")):
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError(f"Video file not found: {video_path}")
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Video has no decodable frames")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate; re-export it at a fixed FPS")
        # shortcut: EOF relies on metadata; use an error-reporting decoder for unindexed footage.
        reported_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        expected_count = int(reported_count) if math.isfinite(reported_count) else 0
        height, width = frame.shape[:2]
        # MP4 encoders require even dimensions; pad rather than drop edge pixels.
        size = (width + width % 2, height + height % 2)
        model = YOLO(str(ensure_model()))
        reader = create_reader()
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix=f"{video_path.stem}_", dir=output_root))
        partial_path = output_dir / "annotated.partial.mp4"
        output_path = output_dir / "annotated.mp4"
        writer = cv2.VideoWriter(str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise OSError("Cannot create MP4 output; check codec support and disk permissions")
        print(f"Saving video in {output_dir.resolve()}")
        frames = 0
        while ok and frame is not None:
            if frame.shape[:2] != (height, width):
                raise ValueError("Video frame dimensions changed during decoding")
            result = model.predict(frame, device="cpu", conf=0.25, verbose=False)[0]
            annotated, _ = annotate_plates(frame, result.boxes.data.tolist(), reader=reader)
            annotated = cv2.copyMakeBorder(
                annotated, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
            writer.write(annotated)
            frames += 1
            if frames % 30 == 0:
                print(f"Processed {frames} frames ({frames / fps:.1f}s of video)")
            ok, frame = capture.read()
        if frames < expected_count:
            raise ValueError(f"Video ended early: decoded {frames} of {expected_count} reported frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    probe = cv2.VideoCapture(str(partial_path))
    try:
        decoded = 0
        while probe.isOpened() and probe.read()[0]:
            decoded += 1
        if decoded != frames:
            raise OSError(f"Output verification failed: {decoded} of {frames} frames readable")
    finally:
        probe.release()
    partial_path.rename(output_path)
    return output_path, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to a recorded vehicle/junction video")
    args = parser.parse_args()
    try:
        output, frames = process_video(args.video)
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. An unfinished .partial.mp4 may remain under outputs/.\n")
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        parser.exit(1, f"Error: {error}. Any .partial.mp4 output is unfinished.\n")
    print(f"Finished {frames} frames. Video: {output.resolve()}")


if __name__ == "__main__":
    main()
