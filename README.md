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

## Next steps

3. Add a detector with license-plate-trained weights and its YOLO dependencies.
4. Connect detection and OCR.
5. Process video and save an annotated demo.
6. Save recognised plates and timestamps to CSV.
7. Evaluate on labelled test footage and write the report.

Cropped-image OCR is implemented. Automatic plate detection and traffic
violation detection are not yet implemented; violations need a defined rule.

## References

- [EasyOCR installation and usage](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Python packages](https://pypi.org/project/opencv-python-headless/)
- [EasyOCR API documentation](https://www.jaided.ai/easyocr/documentation/)
