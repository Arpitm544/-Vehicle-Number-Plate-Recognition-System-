"""Read an already cropped, single-line number plate using CPU OCR."""

import argparse
from pathlib import Path

import cv2
import easyocr


def read_plate(image_path):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f"Image file not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {image_path}. Use a JPG or PNG image.")

    model_dir = Path(__file__).resolve().parent / "models" / "easyocr"
    reader = easyocr.Reader(
        ["en"], gpu=False, detector=False,
        model_storage_directory=str(model_dir),
        user_network_directory=str(model_dir / "user_network"),
    )
    # shortcut: one tightly cropped text line; add text detection for two-line plates.
    results = reader.recognize(
        image, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", detail=1,
    )
    return [("".join(text.upper().split()), float(score))
            for _, text, score in results if text.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a cropped single-line plate image")
    args = parser.parse_args()
    try:
        results = read_plate(args.image)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    if not results:
        parser.exit(1, "No text recognised. Try a clearer, tightly cropped plate image.\n")
    for text, confidence in results:
        print(f"Plate text: {text} | OCR confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()
