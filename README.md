# SatDet - Satellite Object Detection

Satellite images contain huge areas, tiny objects, and complex scenes. SatDet is built to bridge the gap between raw high-resolution imagery and practical object detection for infrastructure monitoring, maritime surveillance, and disaster response.

This project solves three core challenges:

- Large-scale satellite images that cannot be processed directly by standard object detectors.
- DOTA-style rotated polygon labels that must be converted into YOLO-compatible horizontal boxes.
- The need for a complete workflow from dataset preparation and training to inference, API deployment, and interactive visualization.

SatDet uses a YOLOv8s backbone for speed and deployability, tiles large images into 640x640 crops, trains on the processed dataset, and exposes inference through scripts, FastAPI, and a Streamlit demo.

## Important Paths

```text
models/weights/best.pt                 Final trained model for inference/export
models/weights/last.pt                 Last checkpoint, useful for resuming training
data/processed/tiled/                  Dataset used for training and validation
configs/dataset.yaml                   Dataset paths and class names
configs/training.yaml                  Model choice and training hyperparameters
src/training/trainer.py                Training scriptresults/eval/                          Lightweight evaluation output (metrics and class AP plots)results/satdet_v1-6/                   Training results, plots, metrics, and original checkpoints
```

The model was trained from the YOLOv8s pretrained checkpoint configured in [configs/training.yaml](configs/training.yaml):
python -m src.evaluation.evaluator --weights best.pt --data configs/dataset.yaml
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
results/satdet_v1-6
```

> Note: there are three different result folders in this project.

> - `results/satdet_v1-6` contains training/evaluation artifacts, plots, and saved checkpoints.
> - `results/eval` contains lightweight evaluation outputs such as metrics and per-class AP plots.
> - `results/inference` is reserved for new batch prediction outputs from `src.inference.batch_predictor`.

## Results Summary

Final validation metrics after 20 epochs (saved under `results/satdet_v1-6`):

```text
precision:   0.629
recall:      0.556
mAP50:       0.544
mAP50-95:    0.284
```

These values reflect the model performance on the validation dataset using the project evaluation pipeline.

Key evaluation artifacts in `results/satdet_v1-6`:

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
results/satdet_v1-6/weights/best.pt
```

If you want to compare with a fresh inference run, use the evaluation command and inspect the generated plots in `results/satdet_v1-6`.

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
