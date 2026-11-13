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

## Next steps

2. Read text from a clear, manually cropped plate image.
3. Add a detector with license-plate-trained weights and its YOLO dependencies.
4. Connect detection and OCR.
5. Process video and save an annotated demo.
6. Save recognised plates and timestamps to CSV.
7. Evaluate on labelled test footage and write the report.

Plate recognition is not yet implemented. Traffic violation detection requires
a separately defined rule and is not included in Step 1.

## References

- [EasyOCR installation and usage](https://github.com/JaidedAI/EasyOCR)
- [OpenCV Python packages](https://pypi.org/project/opencv-python-headless/)
