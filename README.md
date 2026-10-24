# Pork rasher tray inspection with YOLO

Detect and localise packaging and product-placement defects in pork rasher trays using YOLOv8 object detection, give each tray a PASS/FAIL verdict, and evaluate whether the method is accurate and fast enough for real-time inspection.

Individual project for **MMA3001 Numerical Methods and Machine Learning**, Monash University.
Author: [YOUR NAME], [STUDENT ID]

## Problem and scope

Food-packaging lines rely on manual inspection to catch defective trays before distribution. This project trains an object detector on images of packaged pork rashers and turns its detections into a tray-level decision.

- **Input:** one RGB image of a tray (JPG or PNG, 720 x 540 px in this dataset).
- **Output:** a list of detections (class, confidence, box in pixels) and a tray verdict. A tray **FAILS** if any detection has confidence at or above a threshold, otherwise it **PASSES**.
- **Classes:** `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`. Every class is a defect, so there is no "good" class.
- **Out of scope:** defect severity, fat percentage, counting rashers, segmentation, training a network from scratch.

## Headline results (test split)

The final model is **YOLOv8n at 640 px** (`v8n_640`), selected by validation mAP50-95 before the test split was examined. All three runs are reported below even where another run scored higher on test.

| Run | mAP50 | mAP50-95 | mAP50, packaging-error + unsealed | Mean ms / image | FPS | Params (M) | GFLOPs | Size (MB) |
|---|---|---|---|---|---|---|---|---|
| `v8n_640` (selected) | 0.534 | 0.234 | 0.616 | 8.64 | 116 | 3.14 | 8.75 | 6.50 |
| `v8s_640` | 0.567 | 0.268 | 0.634 | 13.04 | 77 | 11.14 | 28.65 | 22.51 |
| `v8n_480` | 0.544 | 0.229 | 0.709 | 7.95 | 126 | 3.14 | 4.92 | 6.48 |

Speed was measured on a Colab T4 GPU with images preloaded in memory, so it includes resize, inference and NMS but excludes disk reading. The T4 is shared, so expect some variation between sessions. All three models are well above a 30 FPS target.

**Tray-level PASS/FAIL** (`v8n_640`, confidence threshold 0.35 chosen on the validation split; 79 test trays, 68 defective and 11 good; 95% Wilson intervals):

| Measure | Result | 95% CI |
|---|---|---|
| Defective trays rejected (any detection >= 0.35) | 58 / 68 = 0.853 | 0.750 to 0.918 |
| Defective trays with a detection on a real defect (IoU >= 0.3) | 54 / 68 = 0.794 | 0.684 to 0.873 |
| Defective trays with a detection on a real defect (IoU >= 0.5) | 52 / 68 = 0.765 | 0.651 to 0.850 |
| Good trays wrongly rejected | 1 / 11 = 0.091 | 0.016 to 0.377 |

Raw accuracy (0.861) equals that of an "always FAIL" baseline because 86% of test trays are defective, so balanced accuracy (0.881 vs 0.500) is the fairer comparison. Ten defective trays were passed, so this model is better suited to screening than to replacing manual inspection. See the report for limitations.

## Dataset

**Pork Rasher Error (Packaging)**, provided by a Roboflow user, licensed **CC BY 4.0**.
https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging
(Dataset 3 in the MMA3001 project brief.)

The images are **not** stored in this repository. To obtain them, download the YOLOv8-format export (Version 4) from the link above and keep the zip on Google Drive.

| Split | Images as downloaded | After leakage cleaning |
|---|---|---|
| train | 3349 | 3313 |
| valid | 120 | 120 |
| test | 80 | 79 |

**Data leakage.** Roboflow names augmented copies `<original>.rf.<hash>.jpg`. Twelve original frames appeared in train and also in valid (8) or test (4), and one frame appeared in both valid and test. `src/clean_splits.py` removes train copies of any frame present in valid or test, and removes a valid/test shared frame from test (36 train images and 1 test image removed).

**Known data issues.** Class counts are heavily imbalanced (`packaging-error` and `unsealed` dominate). `wrinkle` has no boxes in valid or test, so it cannot be evaluated. `loose-meat` and `twisted-meat` have only 3 to 4 boxes each in valid and test, so their per-class scores are indicative only. Test contains only 11 good trays after cleaning.

## Repository layout

```
README.md
LICENSE                       MIT, covers this project's code only
notebooks/01_pipeline.ipynb   Full workflow: setup, data, training, evaluation, failure analysis
src/clean_splits.py           Detect and remove train/valid/test leakage
src/tray_inspector.py         Tray-level PASS/FAIL logic, input checks, Wilson interval
tests/                        pytest tests (conftest.py, test_*.py) and report.html
docs/                         HTML code documentation generated with pdoc
results/                      CSV tables and figures produced by the notebook
```

Trained weights, run logs and the dataset are not committed (see `.gitignore`). They were saved to Google Drive.

## How to reproduce

The workflow was developed on **Google Colab with a T4 GPU** (Python 3.13, torch 2.11, Ultralytics 8.4.184 for run 1 and 8.4.185 for runs 2 and 3).

1. Download the dataset zip (see above) and place it at `MyDrive/mma3001/dataset3.zip` in Google Drive.
2. Open `notebooks/01_pipeline.ipynb` in Colab and set the runtime to **T4 GPU**.
3. Run **sections 0 and 1** to install packages (`pip install ultralytics pytest pytest-html pdoc`), mount Drive, clone this repo, unzip the data, remove leakage and write the data config. Section 0.3 asks for a GitHub token only if the repo is not already cloned.
4. Run **section 2** to train. Settings shared by all runs: 50 epochs, batch 16, seed 0, `cache='disk'`, early-stopping patience 15. Runs differ only in model size and image size:

   | Run | Model | Image size |
   |---|---|---|
   | `v8n_640` | YOLOv8n | 640 |
   | `v8s_640` | YOLOv8s | 640 |
   | `v8n_480` | YOLOv8n | 480 |

   Finished runs are skipped and interrupted runs resume. Outputs are written to `MyDrive/mma3001/runs/`.
5. Run **section 3** to evaluate on the test split, benchmark speed, choose the tray-level threshold on the validation split and produce the failure analysis. Results are saved to `results/`.

Training is not bit-for-bit deterministic. Each run used a single seed, so small differences between runs should not be over-interpreted.

## Using the tray inspector

```python
import sys
sys.path.insert(0, "src")
from ultralytics import YOLO
from tray_inspector import inspect_tray

model = YOLO("path/to/best.pt")        # weights from your own training run
result = inspect_tray("tray.jpg", model, threshold=0.35)
print(result["verdict"])               # "PASS" or "FAIL"
print(result["detections"])            # [{"class": ..., "confidence": ..., "box_xyxy": [...]}, ...]
```

**Inputs:** path to a `.jpg`, `.jpeg` or `.png` image; a loaded Ultralytics YOLO model; a confidence threshold in (0, 1] (default 0.25; 0.35 was chosen on validation data).
**Invalid input:** a missing file raises `FileNotFoundError`. An unsupported extension, an unreadable image or a threshold outside (0, 1] raises `ValueError`.

## Tests and documentation

```
pytest tests -v --html=tests/report.html --self-contained-html
pdoc src/clean_splits.py src/tray_inspector.py -o docs
```

The 25 tests cover leakage detection and removal (including that it can be re-run safely), the PASS/FAIL rule and its boundary, the Wilson interval, and invalid inputs. They use a **fake model** in place of YOLO, so they verify the surrounding logic and are **not** an end-to-end test of detection quality. The test report is at `tests/report.html` and the code documentation starts at `docs/index.html`.

## Limitations

- Small validation (120) and test (79) sets, only 11 good test trays, and a single training seed. Differences of a few mAP points between the three runs are within the noise, and the false-alarm rate has a wide confidence interval.
- Rare classes (`wrinkle`, `loose-meat`, `twisted-meat`) cannot be evaluated reliably.
- All images come from one setup and are frames with a red overlay (`frame_with_red_...`), so the model may not transfer to raw camera images or other lines.
- The tray rule ignores where a detection is, so a tray can be rejected for a detection that is not on the labelled defect (4 of the 58 rejected defective trays at IoU 0.3).
- Validation scores are slightly optimistic because the best epoch of each run and the tray threshold were both chosen on the validation split.

## Licences and attribution

- **Code:** MIT (see `LICENSE`).
- **Dataset:** CC BY 4.0, Pork Rasher Error (Packaging), Roboflow (link above). Please cite it if you reuse the data.
- **Ultralytics YOLO** is licensed under AGPL-3.0 (with a separate commercial licence). That is fine for coursework, but it matters if the code were deployed commercially.

## AI use

AI assistance (Claude, by Anthropic) was used throughout this project for code drafting, debugging and planning. What it was used for, how its output was checked and the errors it made are described in the AI reflection section of the project report.
