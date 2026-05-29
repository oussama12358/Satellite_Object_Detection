"""
SatDet Training Orchestrator
==============================
Main training script for YOLOv8 on satellite imagery datasets.

Design Decisions:
    1. Uses Ultralytics' native training API for best GPU utilization
    2. MLflow tracking wrapped around training lifecycle
    3. Custom callbacks for aerospace-relevant metrics logging
    4. Gradient clipping and mixed precision for stability with large images
"""

import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

import yaml
import torch
from ultralytics import YOLO
from loguru import logger

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Training metrics will not be tracked remotely.")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_mlflow(experiment_name: str, tracking_uri: str, run_name: str):
    """Initialize MLflow experiment and start run."""
    if not MLFLOW_AVAILABLE:
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow run started: {run.info.run_id}")
        return run
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Continuing without tracking.")
        return None


def log_training_config(config: dict, mlflow_run=None):
    """Log hyperparameters to MLflow."""
    if mlflow_run is None or not MLFLOW_AVAILABLE:
        return
    flat_params = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for k, v in values.items():
                if not isinstance(v, dict):
                    flat_params[f"{section}.{k}"] = v
        else:
            flat_params[section] = values
    mlflow.log_params(flat_params)


def build_training_args(config: dict) -> dict:
    """
    Build Ultralytics training argument dict from our config YAML.

    Maps our structured config → flat Ultralytics kwargs.
    """
    train_cfg = config.get("training", {})
    opt_cfg = config.get("optimizer", {})
    loss_cfg = config.get("loss", {})
    aug_cfg = config.get("augmentation", {})
    out_cfg = config.get("output", {})
    log_cfg = config.get("logging", {})

    args = {
        # Training
        "epochs": train_cfg.get("epochs", 100),
        "patience": train_cfg.get("patience", 20),
        "batch": train_cfg.get("batch", 16),
        "imgsz": train_cfg.get("imgsz", 640),
        "workers": train_cfg.get("workers", 8),
        "device": train_cfg.get("device", "0"),
        "amp": train_cfg.get("amp", True),
        "seed": train_cfg.get("seed", 42),

        # Optimizer
        "optimizer": opt_cfg.get("name", "AdamW"),
        "lr0": opt_cfg.get("lr0", 0.001),
        "lrf": opt_cfg.get("lrf", 0.01),
        "momentum": opt_cfg.get("momentum", 0.937),
        "weight_decay": opt_cfg.get("weight_decay", 0.0005),
        "warmup_epochs": opt_cfg.get("warmup_epochs", 3.0),
        "warmup_momentum": opt_cfg.get("warmup_momentum", 0.8),
        "warmup_bias_lr": opt_cfg.get("warmup_bias_lr", 0.1),

        # Loss weights
        "box": loss_cfg.get("box", 7.5),
        "cls": loss_cfg.get("cls", 0.5),
        "dfl": loss_cfg.get("dfl", 1.5),

        # Augmentation
        "hsv_h": aug_cfg.get("hsv_h", 0.015),
        "hsv_s": aug_cfg.get("hsv_s", 0.7),
        "hsv_v": aug_cfg.get("hsv_v", 0.4),
        "degrees": aug_cfg.get("degrees", 45.0),
        "translate": aug_cfg.get("translate", 0.1),
        "scale": aug_cfg.get("scale", 0.5),
        "flipud": aug_cfg.get("flipud", 0.5),
        "fliplr": aug_cfg.get("fliplr", 0.5),
        "mosaic": aug_cfg.get("mosaic", 1.0),
        "mixup": aug_cfg.get("mixup", 0.1),
        "copy_paste": aug_cfg.get("copy_paste", 0.1),

        # Output
        "project": out_cfg.get("project", "results/train"),
        "name": out_cfg.get("name", "satdet"),
        "exist_ok": out_cfg.get("exist_ok", False),
        "plots": log_cfg.get("plots", True),
        "save_period": log_cfg.get("save_period", 10),
        "verbose": True,
    }
    return args


def find_latest_checkpoint(search_root: str = "results") -> Optional[Path]:
    """Find the newest Ultralytics last.pt checkpoint under the results directory."""
    candidates = sorted(
        Path(search_root).rglob("last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resume_from_checkpoint(checkpoint_path: Path) -> dict:
    """Resume an interrupted Ultralytics training run from last.pt."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Resuming training from: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    results = model.train(resume=True)

    metrics = getattr(results, "results_dict", {}) or {}
    return {
        "checkpoint": str(checkpoint_path),
        "metrics": metrics,
    }


class SatDetTrainer:
    """
    High-level training orchestrator for SatDet.

    Wraps Ultralytics YOLO with:
    - MLflow experiment tracking
    - Automatic hardware detection and optimization
    - Checkpoint management
    - Post-training evaluation summary
    """

    def __init__(self, config_path: str, dataset_config: str):
        self.config = load_config(config_path)
        self.dataset_config = dataset_config
        self.mlflow_run = None
        self._setup_logging()

    def _setup_logging(self):
        logger.remove()
        logger.add(sys.stdout, level="INFO", colorize=True,
                   format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
        logger.add("logs/training_{time}.log", level="DEBUG", rotation="100 MB")

    def _check_hardware(self):
        """Log hardware info for reproducibility records."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name} | {props.total_memory / 1e9:.1f} GB VRAM")
        else:
            logger.warning("No GPU detected. Training will be slow on CPU.")

    def _get_model_weights(self) -> str:
        model_cfg = self.config.get("model", {})
        weights = model_cfg.get("weights", "yolov8s.pt")
        architecture = model_cfg.get("architecture", "yolov8s")

        # If weights path exists locally, use it; otherwise use architecture name
        if Path(weights).exists():
            logger.info(f"Loading weights from: {weights}")
            return weights
        logger.info(f"Using pretrained architecture: {architecture} (COCO pretrained)")
        return f"{architecture}.pt"

    def train(self) -> dict:
        """Execute full training pipeline."""
        self._check_hardware()

        # Setup MLflow
        log_cfg = self.config.get("logging", {}).get("mlflow", {})
        if log_cfg.get("enabled", False) and MLFLOW_AVAILABLE:
            self.mlflow_run = setup_mlflow(
                experiment_name=log_cfg.get("experiment", "satdet"),
                tracking_uri=log_cfg.get("tracking_uri", "http://localhost:5000"),
                run_name=f"yolov8_{time.strftime('%Y%m%d_%H%M%S')}",
            )
            log_training_config(self.config, self.mlflow_run)

        # Build model
        weights = self._get_model_weights()
        model = YOLO(weights)
        logger.info(f"Model loaded: {weights}")

        # Build training args
        train_args = build_training_args(self.config)
        train_args["data"] = self.dataset_config

        # Log dataset config path
        if self.mlflow_run and MLFLOW_AVAILABLE:
            mlflow.log_artifact(self.dataset_config)
            mlflow.log_artifact("configs/training.yaml")

        logger.info("🚀 Starting training...")
        logger.info(f"   Epochs:     {train_args['epochs']}")
        logger.info(f"   Batch size: {train_args['batch']}")
        logger.info(f"   Image size: {train_args['imgsz']}")
        logger.info(f"   Device:     {train_args['device']}")
        logger.info(f"   AMP:        {train_args['amp']}")

        # Train
        results = model.train(**train_args)

        # Log final metrics to MLflow
        if self.mlflow_run and MLFLOW_AVAILABLE:
            try:
                metrics = results.results_dict
                mlflow.log_metrics({
                    "mAP50": metrics.get("metrics/mAP50(B)", 0),
                    "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0),
                    "precision": metrics.get("metrics/precision(B)", 0),
                    "recall": metrics.get("metrics/recall(B)", 0),
                })
                # Log best weights artifact
                best_weights = Path(train_args["project"]) / train_args["name"] / "weights/best.pt"
                if best_weights.exists():
                    mlflow.log_artifact(str(best_weights), artifact_path="weights")
            except Exception as e:
                logger.warning(f"MLflow metrics logging failed: {e}")
            finally:
                mlflow.end_run()

        # Post-training validation
        logger.info("Running post-training validation...")
        val_results = model.val(data=self.dataset_config, verbose=True)

        summary = {
            "best_weights": str(
                Path(train_args["project"]) / train_args["name"] / "weights/best.pt"
            ),
            "map50": val_results.box.map50,
            "map50_95": val_results.box.map,
            "precision": val_results.box.mp,
            "recall": val_results.box.mr,
        }

        # Save summary
        summary_path = Path(train_args["project"]) / train_args["name"] / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.success("✅ Training complete!")
        logger.success(f"   mAP@0.5:      {summary['map50']:.4f}")
        logger.success(f"   mAP@0.5:0.95: {summary['map50_95']:.4f}")
        logger.success(f"   Best weights: {summary['best_weights']}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="Train SatDet YOLOv8 model")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--dataset", default="configs/dataset.yaml")
    parser.add_argument("--model", help="Override model architecture (e.g., yolov8s, yolov8l)")
    parser.add_argument("--epochs", type=int, help="Override epoch count")
    parser.add_argument("--device", help="Override device (e.g., 0, cpu, 0,1)")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest last.pt checkpoint")
    parser.add_argument("--resume-from", help="Path to a specific last.pt checkpoint")
    args = parser.parse_args()

    if args.resume or args.resume_from:
        checkpoint = Path(args.resume_from) if args.resume_from else find_latest_checkpoint()
        if checkpoint is None:
            raise FileNotFoundError("No last.pt checkpoint found under results/.")
        summary = resume_from_checkpoint(checkpoint)
        print(f"\nResumed from: {summary['checkpoint']}")
        return

    trainer = SatDetTrainer(args.config, args.dataset)

    # CLI overrides
    if args.model:
        trainer.config["model"]["architecture"] = args.model
        trainer.config["model"]["weights"] = f"{args.model}.pt"
    if args.epochs:
        trainer.config["training"]["epochs"] = args.epochs
    if args.device:
        trainer.config["training"]["device"] = args.device

    summary = trainer.train()
    print(f"\n🏆 Final mAP@0.5: {summary['map50']:.4f}")
    print(f"🏆 Final mAP@0.5:0.95: {summary['map50_95']:.4f}")


if __name__ == "__main__":
    main()
