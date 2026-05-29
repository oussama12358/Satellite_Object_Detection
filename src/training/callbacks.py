"""
Custom Training Callbacks
===========================
MLflow integration callbacks and aerospace-domain training monitors.

Hooks into Ultralytics callback system:
    on_train_start    → log config and hardware info
    on_train_epoch_end → log per-epoch metrics
    on_val_end        → log validation metrics + confusion matrix
    on_train_end      → log final artifacts
"""

from pathlib import Path
from typing import Optional, Dict

from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowCallback:
    """
    Ultralytics-compatible MLflow logging callback.

    Attach to YOLO model:
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
    """

    def __init__(
        self,
        experiment_name: str = "satdet",
        tracking_uri: str = "http://localhost:5000",
        run_name: Optional[str] = None,
        tags: Optional[Dict] = None,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.tags = tags or {"project": "satdet", "framework": "yolov8"}
        self._run = None
        self._step = 0

    def on_train_start(self, trainer) -> None:
        if not MLFLOW_AVAILABLE:
            return
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._run = mlflow.start_run(
                run_name=self.run_name,
                tags=self.tags,
            )
            # Log hyperparameters
            args = vars(trainer.args) if hasattr(trainer.args, "__dict__") else {}
            safe_params = {
                k: v for k, v in args.items()
                if isinstance(v, (int, float, str, bool)) and k != "device"
            }
            mlflow.log_params(safe_params)
            logger.info(f"MLflow run: {self._run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow on_train_start failed: {e}")

    def on_train_epoch_end(self, trainer) -> None:
        if not MLFLOW_AVAILABLE or self._run is None:
            return
        try:
            metrics = trainer.metrics if hasattr(trainer, "metrics") else {}
            loss_dict = {
                f"train/{k}": float(v)
                for k, v in metrics.items()
                if "loss" in k.lower() and v is not None
            }
            mlflow.log_metrics(loss_dict, step=self._step)
            self._step += 1
        except Exception as e:
            logger.debug(f"MLflow epoch log failed: {e}")

    def on_val_end(self, validator) -> None:
        if not MLFLOW_AVAILABLE or self._run is None:
            return
        try:
            metrics = {}
            if hasattr(validator, "metrics") and validator.metrics:
                m = validator.metrics
                if hasattr(m, "box"):
                    metrics = {
                        "val/mAP50": float(m.box.map50),
                        "val/mAP50-95": float(m.box.map),
                        "val/precision": float(m.box.mp),
                        "val/recall": float(m.box.mr),
                    }
            if metrics:
                mlflow.log_metrics(metrics, step=self._step)
        except Exception as e:
            logger.debug(f"MLflow val log failed: {e}")

    def on_train_end(self, trainer) -> None:
        if not MLFLOW_AVAILABLE or self._run is None:
            return
        try:
            # Log best weights
            save_dir = Path(trainer.save_dir) if hasattr(trainer, "save_dir") else Path("results/train")
            best_pt = save_dir / "weights" / "best.pt"
            if best_pt.exists():
                mlflow.log_artifact(str(best_pt), artifact_path="weights")

            # Log training curves plot
            results_png = save_dir / "results.png"
            if results_png.exists():
                mlflow.log_artifact(str(results_png), artifact_path="plots")

            # Final metrics
            if hasattr(trainer, "metrics"):
                m = trainer.metrics
                final = {
                    f"final/{k}": float(v)
                    for k, v in m.items()
                    if v is not None and isinstance(v, (int, float))
                }
                if final:
                    mlflow.log_metrics(final)

            mlflow.end_run()
            logger.success(f"MLflow run completed: {self._run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow on_train_end failed: {e}")


class EarlyStoppingWithWarmup:
    """
    Custom early stopping that ignores the warmup phase.

    Design: Standard early stopping triggers too aggressively during
    the initial warmup epochs where loss is still high and unstable.
    This callback delays the patience counter until after warmup.
    """

    def __init__(
        self,
        patience: int = 20,
        warmup_epochs: int = 5,
        monitor: str = "val/mAP50",
        mode: str = "max",
    ):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.monitor = monitor
        self.mode = mode
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.counter = 0
        self.epoch = 0

    def __call__(self, trainer) -> bool:
        """
        Returns True if training should stop.
        Attach to on_val_end callback.
        """
        self.epoch += 1

        if self.epoch <= self.warmup_epochs:
            return False

        # Extract monitored metric
        score = None
        if hasattr(trainer, "metrics"):
            metrics = trainer.metrics
            for key in [self.monitor, self.monitor.split("/")[-1]]:
                if key in metrics:
                    score = float(metrics[key])
                    break

        if score is None:
            return False

        improved = (
            (self.mode == "max" and score > self.best_score)
            or (self.mode == "min" and score < self.best_score)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs "
                f"(best {self.monitor}={self.best_score:.4f})"
            )

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered at epoch {self.epoch} "
                f"(no improvement for {self.patience} epochs)"
            )
            return True
        return False


class CheckpointManager:
    """
    Manages model checkpoints with best/last/periodic saves.
    Prunes old checkpoints to save disk space.
    """

    def __init__(
        self,
        save_dir: str,
        keep_top_k: int = 3,
        save_period: int = 10,
    ):
        self.save_dir = Path(save_dir)
        self.keep_top_k = keep_top_k
        self.save_period = save_period
        self._checkpoints: list = []

    def save(self, model, epoch: int, metric: float, tag: str = "periodic") -> None:
        """Save checkpoint and prune old ones."""
        ckpt_path = self.save_dir / f"epoch_{epoch:04d}_{tag}_{metric:.4f}.pt"
        try:
            import torch
            torch.save(model.state_dict(), ckpt_path)
            self._checkpoints.append((metric, epoch, ckpt_path))
            self._checkpoints.sort(key=lambda x: -x[0])  # Sort by metric desc

            # Keep only top-k
            while len(self._checkpoints) > self.keep_top_k:
                _, _, old_path = self._checkpoints.pop()
                if old_path.exists():
                    old_path.unlink()
                    logger.debug(f"Pruned checkpoint: {old_path.name}")
        except Exception as e:
            logger.warning(f"Checkpoint save failed: {e}")
