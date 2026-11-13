# Vehicle Number Plate Recognition System

A computer vision semester project to detect vehicle number plates and read
their text using Python, OpenCV, and EasyOCR.

## Step 1: Environment setup

Use Python 3.11 or 3.12. Python 3.11 is used for local verification.
No Google Colab or GPU is required for this step.

### macOS / Linux

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_setup.py
```

If Python 3.12 is installed instead, replace `python3.11` with `python3.12`.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python check_setup.py
```

The check imports the libraries, processes a small image with OpenCV, and runs
a PyTorch calculation. It should end with:

```text
Setup passed. Ready for Step 2: read a cropped plate image.
```

EasyOCR installs PyTorch as a dependency. Installation needs an internet
connection and may take a few minutes. This step does not download OCR weights
or run recognition; EasyOCR downloads its weights when a Reader is first created
in Step 2.

OpenCV's headless package supports image processing and saved video output,
but does not open desktop windows with `cv2.imshow`. Keep just this OpenCV
package installed to avoid conflicting `cv2` installations.

## Step 2: Read a cropped plate image

Activate the environment from Step 1, then run:

```sh
python read_plate.py samples/synthetic_plate.png
python read_plate.py "path/to/cropped_plate.jpg"
```

The included sample is generated text, not a real vehicle photograph.

Use a clear JPG or PNG containing only one horizontal line of plate characters.
Crop away the vehicle and any small country labels. Full vehicle photographs
and two-line plates need detection/segmentation in later steps.

The first run downloads English OCR recognition weights into `models/easyocr/`.
Internet access is required for that first run; later runs reuse the local model.
Processing runs on CPU. Model files are ignored by Git.

Verified output for the included synthetic sample on the local CPU:

```text
Plate text: MH12AB1234 | OCR confidence: 0.99
```

Confidence is the model's score, not measured accuracy or proof of a valid plate.
This step restricts output to uppercase Latin letters and digits. It does not
correct ambiguous characters such as O/0 or validate regional plate formats.

Run the offline checks with:

```sh
python -m unittest -v
```

These tests mock OCR to check image validation and result handling. Evaluate
recognition separately using real labelled plate images.

## Step 3: Detect plates in a full image

Update the environment, then provide a vehicle or junction image:

```sh
python -m pip install -r requirements.txt
python detect_plates.py "path/to/vehicle.jpg"
```

The first run downloads the 5.47 MB YOLO11n plate detector into `models/`.
The source revision and SHA-256 checksum are pinned in `detect_plates.py`;
subsequent runs check and reuse the local file. The script runs on CPU.

Each run creates a new folder under `outputs/` containing:

- `detected.png`: the full image with green plate boxes and detection scores.
- `plate_1.png`, `plate_2.png`, etc.: unannotated crops for the OCR step.

Runs use separate output folders to avoid overwriting earlier results. A run
with no detections still saves the full image and prints a message. The fixed
minimum detection confidence is 0.25; this is not measured accuracy.

For now, OCR remains a separate command. For a single-line detected plate:

```sh
python read_plate.py "outputs/<run-folder>/plate_1.png"
```

The OCR sample from Step 2 is just text, not a full vehicle scene; use an actual
vehicle image for detection. Two-line crops still need the later OCR improvements.
The offline tests cover crop boundaries, empty detections, invalid images,
non-overwriting outputs, and model download/cache checksum handling.

### Detector source and limitations

The detector is the nano model published by
[morsetechlab](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
revision `251a30d7daedca065f56e04b0af04052c907c68f`. The weights and Ultralytics
are published under AGPL-3.0. Their original licenses apply; model weights are
not committed to this repository.

The model author reports train/test contamination in the upstream dataset.
Do not use the model card's accuracy numbers in this project's report. Evaluate
on independent labelled images/video, especially for Indian plates, motorcycles,
small plates, and low light. This step is a prototype, not an accuracy evaluation.

`ultralytics-opencv-headless` supplies the same `ultralytics` Python API while
using the existing headless OpenCV package. Do not install the regular
`ultralytics` package alongside it. NumPy and OpenCV were updated together to
satisfy the YOLO package's macOS compatibility requirements.

Local smoke test: the detector found one plate (score 0.73) in the public
[FastALPR vehicle sample](https://github.com/ankandrew/fast-alpr/blob/master/assets/test_image.png).
The image was used locally and is not included in this repository. This is a
functionality check, not an independent accuracy benchmark.

## Next steps

4. Connect detection and OCR.
5. Process video and save an annotated demo.
6. Save recognised plates and timestamps to CSV.
7. Evaluate on labelled test footage and write the report.

Cropped-image OCR and image-based plate detection are implemented separately.
Automatic detection-to-OCR integration, video processing, and traffic violation
detection are not yet implemented; violations need a defined rule.

## References

- [EasyOCR installation and usage](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Python packages](https://pypi.org/project/opencv-python-headless/)
- [EasyOCR API documentation](https://www.jaided.ai/easyocr/documentation/)
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/)
