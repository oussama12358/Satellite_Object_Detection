"""
SAHI-Based Sliced Inference
=============================
Implements Slicing Aided Hyper Inference (SAHI) for small object detection
in large satellite images. Reference: arxiv.org/abs/2202.06934

Design Decision:
    Direct inference on 4000×4000 satellite imagery causes small objects
    (vehicles at <10px) to be missed entirely — they're sub-pixel after
    downsampling to 640×640. SAHI solves this by:
    1. Slicing the image into overlapping 640×640 tiles
    2. Running YOLOv8 inference on each tile
    3. Merging tile predictions back to full image coordinates
    4. Applying Weighted Box Fusion (WBF) to handle overlapping detections

    Tradeoff: Latency increases ~10-40× depending on tile count.
    For real-time aerospace requirements, use standard inference on pre-tiled data.
    SAHI is appropriate for batch analysis and reconnaissance workflows.
"""

import time
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger


def as_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor or array-like value to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def weighted_box_fusion(
    boxes_list: List[np.ndarray],
    scores_list: List[np.ndarray],
    labels_list: List[np.ndarray],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified Weighted Box Fusion implementation.
    For production, use the ensemble_boxes library WBF implementation.

    Args:
        boxes_list: List of [N, 4] arrays with normalized [x1,y1,x2,y2]
        scores_list: List of [N] confidence arrays
        labels_list: List of [N] class label arrays
        iou_thr: IoU threshold for grouping boxes
        skip_box_thr: Minimum confidence to include

    Returns:
        (fused_boxes, fused_scores, fused_labels)
    """
    if not boxes_list:
        return np.array([]), np.array([]), np.array([])

    # Flatten all predictions
    all_boxes = np.concatenate(boxes_list, axis=0)
    all_scores = np.concatenate(scores_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Filter low confidence
    mask = all_scores >= skip_box_thr
    all_boxes = all_boxes[mask]
    all_scores = all_scores[mask]
    all_labels = all_labels[mask]

    if len(all_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort by confidence descending
    order = np.argsort(-all_scores)
    all_boxes = all_boxes[order]
    all_scores = all_scores[order]
    all_labels = all_labels[order]

    # Greedy NMS-like WBF grouping
    used = np.zeros(len(all_boxes), dtype=bool)
    fused_boxes, fused_scores, fused_labels = [], [], []

    for i in range(len(all_boxes)):
        if used[i]:
            continue
        # Find all boxes IoU > threshold with same class
        group_mask = (~used) & (all_labels == all_labels[i])
        group_idxs = np.where(group_mask)[0]

        # Compute IoU between box[i] and group
        ious = compute_iou_batch(all_boxes[i], all_boxes[group_idxs])
        close_group = group_idxs[ious >= iou_thr]

        # Weighted fusion
        w = all_scores[close_group]
        fused_box = np.average(all_boxes[close_group], axis=0, weights=w)
        fused_score = np.mean(w)

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)
        fused_labels.append(all_labels[i])
        used[close_group] = True

    if not fused_boxes:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(fused_boxes),
        np.array(fused_scores),
        np.array(fused_labels),
    )


def compute_iou_batch(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and array of boxes. [x1,y1,x2,y2] format."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter + 1e-8

    return inter / union


class SAHIPredictor:
    """
    Sliced Aided Hyper Inference engine for satellite imagery.

    Usage:
        predictor = SAHIPredictor("models/weights/best.pt", conf=0.25)
        results = predictor.predict("satellite_image.jpg")
        predictor.visualize(results, "output.jpg")
    """

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        iou: float = 0.45,
        tile_size: int = 640,
        overlap: float = 0.2,
        wbf_iou_thr: float = 0.55,
        device: Optional[str] = None,
    ):
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.tile_size = tile_size
        self.overlap = overlap
        self.wbf_iou_thr = wbf_iou_thr
        self.device = device or ("cuda:0" if self._cuda_available() else "cpu")
        logger.info(f"SAHIPredictor initialized | device={self.device} | tile={tile_size}px | overlap={overlap}")

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _generate_slices(
        self, *args, img_h: Optional[int] = None, img_w: Optional[int] = None
    ) -> List[Tuple[int, int, int, int]]:
        """Generate tile coordinates with overlap."""
        if len(args) == 2 and img_h is None and img_w is None:
            img_h, img_w = args
        elif len(args) == 1 and args[0] is self and img_h is not None and img_w is not None:
            pass
        elif args or img_h is None or img_w is None:
            raise TypeError("_generate_slices expects img_h and img_w")

        img_h = int(img_h)
        img_w = int(img_w)
        stride = int(self.tile_size * (1 - self.overlap))
        slices = []
        y = 0
        while y < img_h:
            x = 0
            y2 = min(y + self.tile_size, img_h)
            while x < img_w:
                x2 = min(x + self.tile_size, img_w)
                slices.append((x, y, x2, y2))
                if x2 == img_w:
                    break
                x += stride
            if y2 == img_h:
                break
            y += stride
        return slices

    def predict(
        self,
        image: Union[str, np.ndarray],
        verbose: bool = False,
    ) -> Dict:
        """
        Run SAHI inference on a full satellite image.

        Args:
            image: Path or numpy array (BGR)
            verbose: Print per-tile progress

        Returns:
            Detection dict with keys: boxes, scores, labels, class_names, inference_ms
        """
        t0 = time.time()

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot load image: {image}")
        else:
            img = image.copy()

        img_h, img_w = img.shape[:2]
        slices = self._generate_slices(img_h, img_w)

        if verbose:
            logger.info(f"Image size: {img_w}×{img_h} | Tiles: {len(slices)}")

        all_boxes_norm = []
        all_scores = []
        all_labels = []

        for (x1, y1, x2, y2) in slices:
            tile = img[y1:y2, x1:x2]
            tw, th = x2 - x1, y2 - y1

            # Pad to tile_size
            if tw < self.tile_size or th < self.tile_size:
                pad = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                pad[:th, :tw] = tile
                tile = pad

            results = self.model.predict(
                tile, conf=self.conf, iou=self.iou,
                device=self.device, verbose=False
            )

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = as_numpy(r.boxes.xyxy)
                scores = as_numpy(r.boxes.conf)
                labels = as_numpy(r.boxes.cls).astype(int)

                # Translate tile-local coords to full image coords
                boxes[:, [0, 2]] += x1
                boxes[:, [1, 3]] += y1

                # Clip to image bounds
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

                # Normalize for WBF
                boxes_norm = boxes / np.array([img_w, img_h, img_w, img_h], dtype=np.float32)

                all_boxes_norm.append(boxes_norm)
                all_scores.append(scores)
                all_labels.append(labels)

        # Merge with WBF
        fused_boxes_norm, fused_scores, fused_labels = weighted_box_fusion(
            all_boxes_norm, all_scores, all_labels,
            iou_thr=self.wbf_iou_thr,
        )

        # Denormalize
        if len(fused_boxes_norm) > 0:
            fused_boxes = fused_boxes_norm * np.array([img_w, img_h, img_w, img_h])
        else:
            fused_boxes = np.array([]).reshape(0, 4)

        inference_ms = (time.time() - t0) * 1000
        class_names = self.model.names if hasattr(self.model, "names") else {}

        return {
            "boxes": fused_boxes,              # [N, 4] absolute xyxy
            "scores": fused_scores,            # [N] confidence
            "labels": fused_labels,            # [N] class ids
            "class_names": class_names,
            "num_detections": len(fused_scores),
            "image_size": (img_w, img_h),
            "num_tiles": len(slices),
            "inference_ms": inference_ms,
        }

    def predict_file(self, image_path: str, **kwargs) -> Dict:
        return self.predict(image_path, **kwargs)

    def visualize(
        self,
        result: Dict,
        output_path: Optional[str] = None,
        image: Optional[Union[str, np.ndarray]] = None,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> np.ndarray:
        """Draw detections on image and optionally save."""
        PALETTE = [
            (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
            (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
            (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
            (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
        ]

        if image is not None:
            if isinstance(image, (str, Path)):
                img_vis = cv2.imread(str(image))
            else:
                img_vis = image.copy()
        else:
            img_vis = np.zeros((*result["image_size"][::-1], 3), dtype=np.uint8)

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        class_names = result["class_names"]

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = PALETTE[int(label) % len(PALETTE)]
            cls_name = class_names.get(int(label), str(label))

            cv2.rectangle(img_vis, (x1, y1), (x2, y2), color, thickness)

            label_text = f"{cls_name} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img_vis, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img_vis, label_text, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

        # Add stats overlay
        stats_text = (
            f"Detections: {result['num_detections']} | "
            f"Tiles: {result['num_tiles']} | "
            f"{result['inference_ms']:.0f}ms"
        )
        cv2.putText(img_vis, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if output_path:
            cv2.imwrite(output_path, img_vis)
            logger.info(f"Visualization saved: {output_path}")

        return img_vis
