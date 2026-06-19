# SatDet - Satellite Object Detection

SatDet is a YOLOv8-based satellite object detection project for DOTA-style aerial imagery. It converts rotated DOTA polygon annotations into YOLO-compatible boxes, tiles large satellite images into 640x640 crops, trains a deployable detector, and exposes inference through scripts, FastAPI, and Streamlit.

## Results at a Glance

The current published metrics are **validation-set results from the previous training run**, not final test-set results. They were produced after 20 epochs using `models/weights/best.pt`.

The dataset configuration has now been updated to use a clean train/validation/test layout under `data/splits`, with a smaller validation split and a larger final test split:

```text
train: 70%
val:   10%
test:  20%
```

For a rigorous final report, the model should be retrained on `data/splits/train`, selected with `data/splits/val`, and evaluated once on `data/splits/test`.

```text
validation precision:   0.629
validation recall:      0.556
validation mAP50:       0.544
validation mAP50-95:    0.284
```

Additional lightweight validation evaluation saved in `results/eval/metrics.json` reports similar values:

```text
validation precision mean:   0.650
validation recall mean:      0.566
validation mAP50:            0.558
validation mAP50-95:         0.286
```

Best-performing validation classes include `plane`, `tennis-court`, `ship`, and `storage-tank`. Harder classes include `helicopter`, `bridge`, and `small-vehicle`, mainly because these objects are rarer, smaller, or more visually ambiguous in tiled satellite scenes.

## Validation vs Test Set

The numbers above are reported on the validation split (`val/images`) and are useful for model selection, training diagnostics, and hyperparameter comparison. They should not be presented as final test-set performance.

The current [configs/dataset.yaml](configs/dataset.yaml) defines all three splits:

```yaml
path: data/splits
train: train/images
val:   val/images
test:  test/images
```

The split was created at the source-image level with a fixed seed (`42`) to reduce leakage between train, validation, and test tiles. The split manifest is saved in `data/splits/split_manifest.json`.

Recommended split discipline:

- `train`: used to optimize model weights.
- `val`: used during development for checkpoint selection and error analysis.
- `test`: used once at the end for an unbiased final report. Do not tune hyperparameters on `test` — keep it reserved for the final evaluation only.

Run validation and test evaluations into separate folders so the artifacts stay clearly separated:

```bash
python -m src.evaluation.evaluator --weights models/weights/best.pt --data configs/dataset.yaml --split val --output results/val_eval
```

```bash
python -m src.evaluation.evaluator --weights models/weights/best.pt --data configs/dataset.yaml --split test --output results/satdet_v1-6_test_eval
```

## Inference and Confidence Threshold Analysis

A previous batch inference run was completed on the tiled validation images with a confidence threshold of `0.25`:

```text
input images:       9814
confidence:         0.25
total detections:   61314
average/image:      6.25
SAHI:               false
output folder:      results/inference
```

This threshold is practical for visual inspection because it keeps more candidate detections. For deployment, the threshold should be selected according to the application:

- Lower thresholds such as `0.10-0.25` improve recall and reveal more small or low-confidence objects, but they also increase false positives.
- Medium thresholds such as `0.30-0.50` are a better default for dashboards or manual review because they reduce noisy boxes while keeping many useful detections.
- Higher thresholds such as `0.60+` should be used only when false alarms are expensive, because they can miss small vehicles, bridges, and rare classes.

The confidence sweep below was run on the held-out test split with `models/weights/best.pt`. It shows different operating points rather than replacing the standard low-confidence mAP evaluation:

```text
Threshold   Precision   Recall   mAP50   mAP50-95
0.10        0.707       0.618    0.603   0.326
0.25        0.735       0.595    0.530   0.292
0.50        0.888       0.300    0.283   0.168
```

The sweep confirms the expected trade-off: increasing the confidence threshold improves precision, but it sharply reduces recall. A threshold around `0.25` is a reasonable visual-inspection/default operating point for this model, while `0.50` is more conservative and misses many true objects.

The sweep artifacts are saved under:

```text
results/satdet_v1-6_test_eval/threshold_sweep/
```

Qualitative inspection should compare the prediction images in `results/satdet_v1-6_train&eval` with the label images:

```text
val_batch0_pred.jpg     vs     val_batch0_labels.jpg
val_batch1_pred.jpg     vs     val_batch1_labels.jpg
val_batch2_pred.jpg     vs     val_batch2_labels.jpg
```

Observed validation behavior:

- Large, visually distinctive objects such as planes, ships, storage tanks, and tennis courts are detected more reliably.
- Small dense objects, especially small vehicles, are more sensitive to tiling, overlap, and confidence threshold choice.
- Rare classes such as helicopter and bridge need careful qualitative review because a high precision or recall value can be unstable when few examples are present.
- Some DOTA rotated boxes are converted to horizontal boxes, so localization can look less tight for strongly rotated objects.

Future threshold sweeps can extend this analysis with more confidence values and per-class detection counts to refine the deployment threshold.

This project solves three core challenges:

- Large-scale satellite images that cannot be processed directly by standard object detectors.
- DOTA-style rotated polygon labels that must be converted into YOLO-compatible horizontal boxes.
- The need for a complete workflow from dataset preparation and training to inference, API deployment, and interactive visualization.

SatDet uses a YOLOv8s backbone for speed and deployability, tiles large images into 640x640 crops, trains on the processed dataset, and exposes inference through scripts, FastAPI, and a Streamlit demo.

## Important Paths

```text
models/weights/best.pt                 Final trained model for inference/export
models/weights/last.pt                 Last checkpoint, useful for resuming training
data/splits/                           Active train/validation/test dataset
data/processed/all_tiled/              Combined tiled source used to create data/splits
data/processed/tiled/                  Earlier tiled train/validation folders
configs/dataset.yaml                   Dataset paths and class names
configs/training.yaml                  Model choice and training hyperparameters
src/training/trainer.py                Training script
results/eval/                          Lightweight evaluation output
results/satdet_v1-6_train&eval/        Training results, validation plots, metrics, and original checkpoints
results/satdet_v1-6_test_eval/         Final test metrics, plots, predictions, and confusion matrices
```

The model was trained from the YOLOv8s pretrained checkpoint configured in [configs/training.yaml](configs/training.yaml):

```yaml
model:
  architecture: yolov8s
  weights: yolov8s.pt
```

## Current Model

The trained model is available at:

```text
models/weights/best.pt
```

The full training run kept for metrics and plots is:

```text
results/satdet_v1-6_train&eval
```

> Note: there are three different result folders in this project.

> - `results/satdet_v1-6_train&eval` contains training/validation artifacts, plots, and saved checkpoints.
> - `results/satdet_v1-6_test_eval` contains final test-set evaluation metrics, plots, and predictions.
> - `results/eval` contains lightweight evaluation outputs such as metrics and per-class AP plots.
> - `results/inference` is reserved for new batch prediction outputs from `src.inference.batch_predictor`.

## Validation Results

Final validation metrics after 20 epochs, saved under `results/satdet_v1-6_train&eval`:

```text
precision:   0.629
recall:      0.556
mAP50:       0.544
mAP50-95:    0.284
```

These values reflect model performance on the validation dataset using the project evaluation pipeline. They are not test-set metrics.

Key training/validation artifacts in `results/satdet_v1-6_train&eval`:

- `results.csv` – epoch-by-epoch training and validation metrics
- `args.yaml` – full training arguments and configuration
- `results.png` – training curves summary
- `BoxP_curve.png`, `BoxR_curve.png`, `BoxPR_curve.png`, `BoxF1_curve.png`
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- example predictions:
  - `val_batch0_pred.jpg`, `val_batch1_pred.jpg`, `val_batch2_pred.jpg`
  - `val_batch0_labels.jpg`, `val_batch1_labels.jpg`, `val_batch2_labels.jpg`

### What this means

- `precision` shows how many predicted boxes are correct.
- `recall` shows how many true objects were detected.
- `mAP50` is mean average precision at 0.5 IoU, a standard object detection score.
- `mAP50-95` is the average across stricter IoU thresholds and is a stronger indicator of model quality.

The trained model checkpoint used for inference is:

```text
results/satdet_v1-6_train&eval/weights/best.pt
```

If you want to compare with a fresh test run, use the evaluation command and inspect the generated plots in `results/satdet_v1-6_test_eval`.

## Dataset

The active dataset config is [configs/dataset.yaml](configs/dataset.yaml). It uses DOTA v1.0 with 15 classes:

```text
0 plane
1 ship
2 storage-tank
3 baseball-diamond
4 tennis-court
5 basketball-court
6 ground-track-field
7 harbor
8 bridge
9 large-vehicle
10 small-vehicle
11 helicopter
12 roundabout
13 soccer-ball-field
14 swimming-pool
```

Expected tiled training layout:

```text
data/processed/tiled/
  train/images
  train/labels
  val/images
  val/labels
  test/images   optional held-out split for final evaluation
  test/labels   optional held-out split for final evaluation
```

Raw DOTA data is expected under:

```text
data/raw/dota/DOTA/
  train/images
  train/labelTxt
  val/images
  val/labelTxt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

## Prepare Data

Run this only if the processed tiled dataset does not already exist or you changed the raw data.

Convert DOTA labels to YOLO:

```bash
python -m src.data.dota_converter --input data/raw/dota/DOTA/train --output data/processed/train --version v1.0
python -m src.data.dota_converter --input data/raw/dota/DOTA/val --output data/processed/val --version v1.0
```

Tile images:

```bash
python -m src.data.tiling --input data/processed/train --output data/processed/tiled/train --tile-size 640 --overlap 0.2 --workers 4
python -m src.data.tiling --input data/processed/val --output data/processed/tiled/val --tile-size 640 --overlap 0.2 --workers 4
```

## Train

Training is implemented in:

```text
src/training/trainer.py
```

Start training:

```bash
python -m src.training.trainer --config configs/training.yaml --dataset configs/dataset.yaml
```

Resume from the latest checkpoint:

```bash
python -m src.training.trainer --resume
```

Useful overrides:

```bash
python -m src.training.trainer --config configs/training.yaml --dataset configs/dataset.yaml --model yolov8s --epochs 50 --device 0
python -m src.training.trainer --config configs/training.yaml --dataset configs/dataset.yaml --device cpu
```

## Evaluate

```bash
python -m src.evaluation.evaluator --weights models/weights/best.pt --data configs/dataset.yaml
```

## Inference

Batch inference:

```bash
python -m src.inference.batch_predictor --weights models/weights/best.pt --input data/processed/tiled/val/images --output results/inference
```

Example completed run:

```text
Processing 9814 images from data\processed\tiled\val\images
Batch complete: 9814 images, 61314 total detections
```

The batch output is written to:

```text
results/inference
```

FastAPI service:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/satellite_image.png"
```

Streamlit demo:

```bash
streamlit run src/visualization/streamlit_app.py
```

## Design Decisions

- **YOLOv8s backbone:** chosen for a strong speed/accuracy tradeoff and simple deployment to PyTorch, ONNX, and edge runtimes.
- **640x640 tiling with 20% overlap:** avoids losing small objects when large satellite images are resized directly.
- **DOTA OBB to HBB conversion:** standard YOLOv8 predicts horizontal boxes, so DOTA polygons are converted to axis-aligned boxes.
- **Source-image split discipline:** tiles from the same original image should stay in the same split to avoid validation leakage.
- **SAHI support:** large raw satellite images can be sliced at inference time and merged with Weighted Box Fusion.
- **Evaluation confidence:** mAP should be computed with very low confidence, while deployment can use a practical threshold such as 0.25.
- **FastAPI + Streamlit:** FastAPI is for API deployment; Streamlit is for quick visual inspection.

## Project Structure

```text
configs/                 Dataset and training configuration
data/                    Raw and processed datasets (ignored by git)
docker/                  Dockerfile and docker-compose files
models/                  Optional exported/production model artifacts
results/                 Training and inference outputs (generated)
scripts/                 Helper scripts
src/
  api/                   FastAPI service
  data/                  DOTA/xView conversion, tiling, splitting, stats
  evaluation/            Validation and reporting
  inference/             Standard and SAHI inference
  models/                ONNX export helpers
  training/              Training wrapper and callbacks
  utils/                 Logging and geospatial helpers
  visualization/         Streamlit and Grad-CAM tools
tests/                   Unit and integration tests
```

## Tests

```bash
python -m pytest
```

## Notes

- Keep `data/`, `results/`, and model weights out of git unless a specific artifact is intentionally published.
- `best.pt` is the checkpoint to use for inference.
- `last.pt` is useful if training needs to resume.
