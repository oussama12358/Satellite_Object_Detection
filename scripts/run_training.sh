#!/bin/bash
# =============================================================================
# SatDet Training Launch Script
# Supports single GPU, multi-GPU DDP, and resume from checkpoint
# Usage: bash scripts/run_training.sh [--model yolov8s] [--epochs 100] [--resume]
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-yolov8s}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-16}"
DEVICE="${DEVICE:-0}"
RESUME="${RESUME:-false}"
CONFIG="${CONFIG:-configs/training.yaml}"
DATASET="${DATASET:-configs/dataset.yaml}"
EXP_NAME="${EXP_NAME:-satdet_$(date +%Y%m%d_%H%M%S)}"
WORKERS="${WORKERS:-8}"

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)    MODEL="$2";    shift 2 ;;
        --epochs)   EPOCHS="$2";   shift 2 ;;
        --batch)    BATCH="$2";    shift 2 ;;
        --device)   DEVICE="$2";   shift 2 ;;
        --resume)   RESUME="true"; shift ;;
        --config)   CONFIG="$2";   shift 2 ;;
        --dataset)  DATASET="$2";  shift 2 ;;
        --name)     EXP_NAME="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "
╔══════════════════════════════════════════════════╗
║  SatDet — Training Launch                        ║
╚══════════════════════════════════════════════════╝
  Model:    $MODEL
  Epochs:   $EPOCHS
  Batch:    $BATCH
  Device:   $DEVICE
  Config:   $CONFIG
  Dataset:  $DATASET
  Run name: $EXP_NAME
  Resume:   $RESUME
"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "🔍 Pre-flight checks..."

# Check Python environment
python -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')"
python -c "import torch; print(f'  PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}')"

# Check dataset config exists
if [ ! -f "$DATASET" ]; then
    echo "❌ Dataset config not found: $DATASET"
    echo "   Run scripts/prepare_dataset.sh first."
    exit 1
fi

# Check GPU availability if not using CPU
if [ "$DEVICE" != "cpu" ]; then
    python -c "
import torch
if not torch.cuda.is_available():
    print('⚠️  WARNING: CUDA not available. Falling back to CPU.')
else:
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory/1e9:.1f} GB)')
"
fi

# ── Start MLflow server if configured ────────────────────────────────────────
if command -v mlflow &>/dev/null; then
    echo "📈 Starting MLflow server (background)..."
    mkdir -p mlruns
    mlflow server \
        --host 127.0.0.1 \
        --port 5000 \
        --default-artifact-root ./mlruns \
        --backend-store-uri ./mlruns \
        &>/dev/null &
    MLFLOW_PID=$!
    echo "  MLflow PID: $MLFLOW_PID | UI: http://127.0.0.1:5000"
    sleep 2
fi

# ── Launch Training ───────────────────────────────────────────────────────────
echo "🚀 Starting training..."

EXTRA_ARGS=""
if [ "$RESUME" = "true" ]; then
    # Find the latest checkpoint
    LAST_PT=$(find runs/train -name "last.pt" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -1 | awk '{print $2}')
    if [ -n "$LAST_PT" ]; then
        echo "  Resuming from: $LAST_PT"
        EXTRA_ARGS="--resume"
    else
        echo "  No checkpoint found, starting fresh."
    fi
fi

python -m src.training.trainer \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --epochs "$EPOCHS" \
    --device "$DEVICE" \
    $EXTRA_ARGS

# ── Post-training ─────────────────────────────────────────────────────────────
echo ""
echo "✅ Training complete!"
echo ""
echo "📊 Next steps:"
echo "  1. Evaluate:  python -m src.evaluation.evaluator --weights runs/train/satdet*/weights/best.pt --data $DATASET"
echo "  2. Export:    python -m src.models.onnx_export --weights runs/train/satdet*/weights/best.pt"
echo "  3. Serve API: uvicorn src.api.main:app --host 127.0.0.1 --port 8000"
echo "  4. Demo:      streamlit run src/visualization/streamlit_app.py"
