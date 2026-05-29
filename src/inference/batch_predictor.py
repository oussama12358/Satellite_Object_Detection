"""
Batch Inference Engine
=======================
Process entire folders of satellite imagery with configurable parallelism.
Outputs structured JSON results and optional visualizations.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Any, Optional, List, Dict

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from ultralytics import YOLO


def as_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like value to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


class BatchPredictor:
    """
    Folder-level batch inference with result persistence.

    Supports both standard YOLO inference and SAHI for large images.
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        use_sahi: bool = False,
        tile_size: int = 640,
        overlap: float = 0.2,
        device: Optional[str] = None,
        save_visualizations: bool = True,
        output_format: str = "json",
    ):
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.use_sahi = use_sahi
        self.save_visualizations = save_visualizations
        self.output_format = output_format

        if use_sahi:
            from src.inference.sahi_predictor import SAHIPredictor
            self.predictor = SAHIPredictor(
                weights, conf=conf, iou=iou,
                tile_size=tile_size, overlap=overlap,
                device=device
            )
        else:
            self.model = YOLO(weights)
            self.device = device or "cuda:0"

    def _predict_single_standard(self, image_path: Path) -> Dict:
        """Standard YOLOv8 inference for pre-tiled or small images."""
        t0 = time.time()
        results = self.model.predict(
            str(image_path), conf=self.conf, iou=self.iou,
            device=self.device, verbose=False
        )
        ms = (time.time() - t0) * 1000

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, score, cls in zip(
                as_numpy(r.boxes.xyxy),
                as_numpy(r.boxes.conf),
                as_numpy(r.boxes.cls),
            ):
                x1, y1, x2, y2 = map(float, box)
                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "confidence": float(score),
                    "class_id": int(cls),
                    "class_name": self.model.names.get(int(cls), str(int(cls))),
                })

        return {
            "file": image_path.name,
            "detections": detections,
            "num_detections": len(detections),
            "inference_ms": ms,
        }

    def _predict_single_sahi(self, image_path: Path) -> Dict:
        """SAHI inference for large satellite imagery."""
        result = self.predictor.predict(str(image_path))
        detections = []
        for box, score, label in zip(
            result["boxes"], result["scores"], result["labels"]
        ):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "confidence": float(score),
                "class_id": int(label),
                "class_name": result["class_names"].get(int(label), str(int(label))),
            })
        return {
            "file": image_path.name,
            "detections": detections,
            "num_detections": len(detections),
            "inference_ms": result["inference_ms"],
            "num_tiles": result["num_tiles"],
        }

    def run(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Optional[List[str]] = None,
        max_images: Optional[int] = None,
        num_workers: int = 1,
    ) -> Dict:
        """
        Run batch inference on a directory.

        Args:
            input_dir: Directory with images
            output_dir: Directory for results
            extensions: Image extensions to process
            max_images: Limit number of images (for testing)
            num_workers: Parallel workers (use 1 for GPU inference)

        Returns:
            Batch statistics dict
        """
        extensions = extensions or [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        vis_dir = output_path / "visualizations"
        json_dir = output_path / "detections"
        vis_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)

        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = sorted(image_files)

        if max_images:
            image_files = image_files[:max_images]

        logger.info(f"Processing {len(image_files)} images from {input_path}")

        all_results = []
        total_detections = 0
        predict_fn = self._predict_single_sahi if self.use_sahi else self._predict_single_standard

        for img_path in tqdm(image_files, desc="Batch inference"):
            try:
                result = predict_fn(img_path)
                all_results.append(result)
                total_detections += result["num_detections"]

                # Save per-image JSON
                json_path = json_dir / f"{img_path.stem}.json"
                with open(json_path, "w") as f:
                    json.dump(result, f, indent=2)

                # Save visualization
                if self.save_visualizations and result["num_detections"] > 0:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        self._draw_detections(img, result["detections"])
                        cv2.imwrite(str(vis_dir / f"{img_path.stem}_pred.jpg"), img)

            except Exception as e:
                logger.error(f"Failed on {img_path.name}: {e}")
                all_results.append({"file": img_path.name, "error": str(e)})

        # Save aggregate results
        summary = {
            "total_images": len(image_files),
            "total_detections": total_detections,
            "avg_detections_per_image": total_detections / max(len(image_files), 1),
            "use_sahi": self.use_sahi,
            "conf_threshold": self.conf,
            "results": all_results,
        }

        with open(output_path / "batch_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.success(
            f"Batch complete: {len(image_files)} images, "
            f"{total_detections} total detections"
        )
        return summary

    @staticmethod
    def _draw_detections(img: np.ndarray, detections: List[Dict]) -> None:
        PALETTE = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
        ]
        for det in detections:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
            cls_id = det["class_id"]
            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def main():
    parser = argparse.ArgumentParser(description="Batch satellite image inference")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--sahi", action="store_true")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--no-vis", action="store_true")
    args = parser.parse_args()

    predictor = BatchPredictor(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        use_sahi=args.sahi,
        tile_size=args.tile_size,
        save_visualizations=not args.no_vis,
    )
    predictor.run(args.input, args.output, max_images=args.max_images)


if __name__ == "__main__":
    main()
