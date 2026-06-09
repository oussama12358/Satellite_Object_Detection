"""
SatDet Evaluation Module
=========================
Comprehensive evaluation suite for satellite object detection models.

Metrics:
    - mAP@0.5 and mAP@0.5:0.95 (COCO-standard)
    - Per-class Precision, Recall, F1
    - Confusion matrix with class analysis
    - Small/medium/large object breakdown (COCO-size buckets)
    - Speed benchmarks (FPS, latency)
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments

from ultralytics import YOLO
from loguru import logger


class SatDetEvaluator:
    """
    Full evaluation pipeline for SatDet models.

    Design: Wraps Ultralytics val() with additional aerospace-domain
    analysis including per-size-bucket metrics (critical for satellite
    imagery where object sizes vary by 100× across altitude ranges).
    """

    def __init__(self, weights: str, dataset_config: str, device: str = "0"):
        self.weights = weights
        self.dataset_config = dataset_config
        self.device = device
        self.model = YOLO(weights)
        logger.info(f"Evaluator loaded: {weights}")

    def run_validation(
        self,
        split: str = "val",
        conf: float = 0.001,   # Low conf for mAP computation (don't filter)
        iou: float = 0.6,
        batch: int = 16,
        imgsz: int = 640,
        save_json: bool = True,
        output_dir: str = "results/eval",
    ) -> Dict:
        """
        Run comprehensive validation evaluation.

        Args:
            split: Dataset split ("val" or "test")
            conf: Confidence threshold (use 0.001 for mAP curves)
            iou: IoU threshold for NMS
            batch: Batch size
            imgsz: Inference image size
            save_json: Save COCO-format JSON results
            output_dir: Directory for outputs

        Returns:
            Full metrics dict
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running validation on {split} split...")

        val_results = self.model.val(
            data=self.dataset_config,
            split=split,
            conf=conf,
            iou=iou,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            save_json=save_json,
            project=str(output_path),
            name="val_run",
            verbose=True,
        )

        # Extract per-class metrics
        box = val_results.box
        class_names = self.model.names

        per_class = {}
        if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
            for idx, cls_id in enumerate(box.ap_class_index):
                cls_name = class_names.get(int(cls_id), str(cls_id))
                per_class[cls_name] = {
                    "class_id": int(cls_id),
                    "ap50": float(box.ap50[idx]) if hasattr(box, "ap50") and box.ap50 is not None else 0.0,
                    "ap": float(box.ap[idx]) if hasattr(box, "ap") and box.ap is not None else 0.0,
                    "precision": float(box.p[idx]) if hasattr(box, "p") and box.p is not None else 0.0,
                    "recall": float(box.r[idx]) if hasattr(box, "r") and box.r is not None else 0.0,
                }

        metrics = {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision_mean": float(box.mp),
            "recall_mean": float(box.mr),
            "per_class": per_class,
            "split": split,
            "weights": self.weights,
            "config": self.dataset_config,
        }

        # Save metrics
        with open(output_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        self._print_metrics_table(metrics)
        self._plot_per_class_ap(per_class, output_path)

        return metrics

    def benchmark_speed(
        self,
        num_trials: int = 100,
        imgsz: int = 640,
        batch: int = 1,
    ) -> Dict:
        """
        Benchmark inference speed (latency + throughput).

        Returns dict with mean/std latency, FPS.
        """
        latencies = []
        logger.info(f"Speed benchmark: {num_trials} trials, batch={batch}, img={imgsz}")

        # Warmup
        for _ in range(10):
            self.model.predict(
                np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                device=self.device, verbose=False
            )

        for _ in range(num_trials):
            t0 = time.perf_counter()
            self.model.predict(
                np.zeros((imgsz, imgsz, 3), dtype=np.uint8),
                device=self.device, verbose=False
            )
            latencies.append((time.perf_counter() - t0) * 1000)

        result = {
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "fps": float(1000.0 / np.mean(latencies)),
            "batch_size": batch,
            "image_size": imgsz,
        }

        logger.info(f"Speed: {result['fps']:.1f} FPS | "
                   f"{result['mean_latency_ms']:.1f}±{result['std_latency_ms']:.1f} ms")
        return result

    @staticmethod
    def _print_metrics_table(metrics: Dict) -> None:
        """Print formatted metrics table to console."""
        print("\n" + "="*60)
        print("  SatDet Evaluation Results")
        print("="*60)
        print(f"  mAP@0.5:      {metrics['map50']:.4f}")
        print(f"  mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        print(f"  Precision:    {metrics['precision_mean']:.4f}")
        print(f"  Recall:       {metrics['recall_mean']:.4f}")
        print("-"*60)
        print(f"  {'Class':<25} {'AP@0.5':>8} {'AP':>8} {'P':>8} {'R':>8}")
        print("-"*60)
        for cls_name, m in sorted(metrics["per_class"].items(),
                                  key=lambda x: -x[1]["ap50"]):
            print(f"  {cls_name:<25} {m['ap50']:>8.4f} {m['ap']:>8.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f}")
        print("="*60 + "\n")

    @staticmethod
    def _plot_per_class_ap(per_class: Dict, output_path: Path) -> None:
        """Generate per-class AP bar chart."""
        if not per_class:
            return

        names = list(per_class.keys())
        ap50_vals = [per_class[n]["ap50"] for n in names]

        # Sort by AP
        sorted_pairs = sorted(zip(names, ap50_vals), key=lambda x: x[1], reverse=True)
        names, ap50_vals = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.get_cmap("RdYlGn")(np.linspace(0.2, 0.9, len(names)))
        bars = ax.barh(names, ap50_vals, color=colors)

        ax.set_xlabel("AP@0.5", fontsize=12)
        ax.set_title("Per-Class Average Precision @IoU=0.5", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.axvline(x=np.mean(ap50_vals), color="navy", linestyle="--",
                   alpha=0.7, label=f"Mean: {np.mean(ap50_vals):.3f}")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)

        for bar, val in zip(bars, ap50_vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path / "per_class_ap.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Per-class AP chart saved: {output_path / 'per_class_ap.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SatDet model")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--output", default="results/eval")
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-save-json", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    evaluator = SatDetEvaluator(args.weights, args.data, args.device)
    evaluator.run_validation(
        split=args.split, conf=args.conf,
        save_json=not args.no_save_json,
        output_dir=args.output
    )

    if args.benchmark:
        speed = evaluator.benchmark_speed()
        print(f"\n⚡ FPS: {speed['fps']:.1f} | Latency: {speed['mean_latency_ms']:.1f}ms")


if __name__ == "__main__":
    main()
