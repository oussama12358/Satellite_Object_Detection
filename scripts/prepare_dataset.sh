#!/bin/bash
# =============================================================================
# SatDet Dataset Preparation Pipeline
# Usage: bash scripts/prepare_dataset.sh [--dota-dir PATH] [--classes CLASSES]
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DOTA_DIR="${DOTA_DIR:-data/raw/dota}"
OUTPUT_DIR="${OUTPUT_DIR:-data/processed}"
SPLITS_DIR="${SPLITS_DIR:-data/splits}"
TILE_SIZE="${TILE_SIZE:-640}"
OVERLAP="${OVERLAP:-0.2}"
DOTA_VERSION="${DOTA_VERSION:-v1.0}"
CLASSES="${CLASSES:-plane ship large-vehicle small-vehicle helicopter storage-tank harbor bridge}"
WORKERS="${WORKERS:-4}"

echo "
╔══════════════════════════════════════════════╗
║  SatDet — Dataset Preparation Pipeline       ║
╚══════════════════════════════════════════════╝
  DOTA dir:    $DOTA_DIR
  Output:      $OUTPUT_DIR
  Tile size:   ${TILE_SIZE}px
  Overlap:     ${OVERLAP}
  Version:     $DOTA_VERSION
  Workers:     $WORKERS
"

# ── Step 1: Convert DOTA annotations → YOLO ─────────────────────────────────
echo "Step 1/4: Converting DOTA annotations to YOLO format..."
python -m src.data.dota_converter \
    --input "$DOTA_DIR" \
    --output "$OUTPUT_DIR" \
    --version "$DOTA_VERSION" \
    --skip-difficult \
    --classes $CLASSES

# ── Step 2: Tile large images ─────────────────────────────────────────────────
echo "Step 2/4: Tiling images (${TILE_SIZE}px, ${OVERLAP} overlap)..."
python -m src.data.tiling \
    --input "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/tiled" \
    --tile-size "$TILE_SIZE" \
    --overlap "$OVERLAP" \
    --min-visibility 0.3 \
    --workers "$WORKERS"

# ── Step 3: Create train/val/test splits ──────────────────────────────────────
echo " Step 3/4: Creating stratified splits..."
python -m src.data.dataset_splitter \
    --processed "${OUTPUT_DIR}/tiled" \
    --output "$SPLITS_DIR" \
    --train-ratio 0.70 \
    --val-ratio 0.10 \
    --test-ratio 0.20 \
    --seed 42

# ── Step 4: Statistics report ─────────────────────────────────────────────────
echo "Step 4/4: Generating dataset statistics..."
python -c "
import json
from pathlib import Path
from collections import Counter

splits_dir = Path('$SPLITS_DIR')
for split in ['train', 'val', 'test']:
    lbl_dir = splits_dir / split / 'labels'
    if not lbl_dir.exists():
        continue
    counts = Counter()
    total = 0
    for f in lbl_dir.glob('*.txt'):
        for line in open(f):
            parts = line.strip().split()
            if parts:
                counts[int(parts[0])] += 1
                total += 1
    print(f'  {split:6s}: {len(list(lbl_dir.glob(\"*.txt\"))):5d} images | {total:7d} annotations')
"

echo "
 Dataset preparation complete!
   Splits directory: $SPLITS_DIR
   Update configs/dataset.yaml to point to: $SPLITS_DIR
"
