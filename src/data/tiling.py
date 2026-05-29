"""
Image Tiling Pipeline for Satellite Imagery
============================================
Splits high-resolution satellite images (4000×4000+) into overlapping tiles
for YOLOv8 training and inference.

Design Decision:
    Standard YOLOv8 input is 640×640. A 4000×4000 satellite image downscaled to
    640×640 means each pixel represents ~6.25× the original — tiny objects (ships,
    vehicles at <10px original) become sub-pixel and invisible to the model.

    Strategy: Tile at native resolution with overlap.
    - Tile size: 640×640 (matches model input)
    - Overlap: 20% (128px) — ensures objects near tile borders appear fully in
      at least one tile
    - At inference: use WBF (Weighted Box Fusion) to merge cross-tile detections
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger

YoloAnnotation = Tuple[int, float, float, float, float]


def generate_tile_coords(
    img_width: int,
    img_height: int,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate (x_min, y_min, x_max, y_max) for all tiles covering the image.

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        tile_size: Tile size (square)
        overlap: Fraction of tile_size used as overlap

    Returns:
        List of (x1, y1, x2, y2) tuples
    """
    stride = int(tile_size * (1 - overlap))
    tiles = []

    y = 0
    while y < img_height:
        x = 0
        y2 = min(y + tile_size, img_height)
        while x < img_width:
            x2 = min(x + tile_size, img_width)
            tiles.append((x, y, x2, y2))
            if x2 == img_width:
                break
            x += stride
        if y2 == img_height:
            break
        y += stride

    return tiles


def clip_yolo_bbox_to_tile(
    class_id: int,
    cx: float, cy: float, w: float, h: float,
    tile_x1: int, tile_y1: int, tile_x2: int, tile_y2: int,
    img_width: int, img_height: int,
    min_visibility: float = 0.3,
) -> Optional[YoloAnnotation]:
    """
    Transform and clip a YOLO bounding box to a tile's coordinate space.

    Args:
        class_id, cx, cy, w, h: YOLO format annotation (image-normalized)
        tile_x1/y1/x2/y2: Tile bounds in absolute image pixels
        img_width/img_height: Original image dimensions
        min_visibility: Minimum fraction of object area that must be in tile

    Returns:
        (class_id, new_cx, new_cy, new_w, new_h) or None if filtered out
    """
    # Convert to absolute image coordinates
    abs_cx = cx * img_width
    abs_cy = cy * img_height
    abs_w = w * img_width
    abs_h = h * img_height

    obj_x1 = abs_cx - abs_w / 2
    obj_y1 = abs_cy - abs_h / 2
    obj_x2 = abs_cx + abs_w / 2
    obj_y2 = abs_cy + abs_h / 2

    # Compute intersection with tile
    inter_x1 = max(obj_x1, tile_x1)
    inter_y1 = max(obj_y1, tile_y1)
    inter_x2 = min(obj_x2, tile_x2)
    inter_y2 = min(obj_y2, tile_y2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return None  # No overlap

    # Compute visibility ratio
    obj_area = abs_w * abs_h
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    if obj_area < 1e-6 or inter_area / obj_area < min_visibility:
        return None  # Object too occluded in this tile

    tile_w = tile_x2 - tile_x1
    tile_h = tile_y2 - tile_y1

    # Clip box to tile bounds
    clipped_cx = (inter_x1 + inter_x2) / 2 - tile_x1
    clipped_cy = (inter_y1 + inter_y2) / 2 - tile_y1
    clipped_w = inter_x2 - inter_x1
    clipped_h = inter_y2 - inter_y1

    # Normalize to tile space
    norm_cx = clipped_cx / tile_w
    norm_cy = clipped_cy / tile_h
    norm_w = clipped_w / tile_w
    norm_h = clipped_h / tile_h

    # Clamp
    norm_cx = float(np.clip(norm_cx, 0, 1))
    norm_cy = float(np.clip(norm_cy, 0, 1))
    norm_w = float(np.clip(norm_w, 0, 1))
    norm_h = float(np.clip(norm_h, 0, 1))

    if norm_w < 0.001 or norm_h < 0.001:
        return None

    return (class_id, norm_cx, norm_cy, norm_w, norm_h)


def tile_image_and_labels(
    image_path: Path,
    label_path: Optional[Path],
    output_images_dir: Path,
    output_labels_dir: Path,
    tile_size: int = 640,
    overlap: float = 0.2,
    min_visibility: float = 0.3,
    save_empty_tiles: bool = False,
) -> dict:
    """
    Tile a single image and its YOLO labels.

    Returns stats dict with tile counts.
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Cannot load image: {image_path}")
        return {"tiles_created": 0, "tiles_skipped": 0}

    img_height, img_width = img.shape[:2]
    stem = image_path.stem

    # Parse labels
    annotations: List[YoloAnnotation] = []
    if label_path and label_path.exists():
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:])
                    annotations.append((class_id, cx, cy, w, h))

    # Generate tile coordinates
    tiles = generate_tile_coords(img_width, img_height, tile_size, overlap)

    tiles_created = 0
    tiles_skipped = 0

    for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
        # Pad tile to exact tile_size if at image boundary
        tile_img = img[y1:y2, x1:x2]
        actual_h, actual_w = tile_img.shape[:2]

        if actual_w < tile_size or actual_h < tile_size:
            # Zero-pad boundary tiles to maintain consistent input size
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:actual_h, :actual_w] = tile_img
            tile_img = padded

        # Clip annotations to this tile
        tile_annotations: List[YoloAnnotation] = []
        for ann in annotations:
            class_id, cx, cy, w, h = ann
            clipped = clip_yolo_bbox_to_tile(
                class_id, cx, cy, w, h,
                x1, y1, x2, y2,
                img_width, img_height,
                min_visibility,
            )
            if clipped is not None:
                tile_annotations.append(clipped)

        # Skip empty tiles if not needed (saves disk space significantly)
        if not tile_annotations and not save_empty_tiles:
            tiles_skipped += 1
            continue

        # Save tile
        tile_name = f"{stem}__tile_{tile_idx:04d}"
        img_out = output_images_dir / f"{tile_name}.jpg"
        cv2.imwrite(
            str(img_out), tile_img,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        # Save labels
        lbl_out = output_labels_dir / f"{tile_name}.txt"
        with open(lbl_out, "w") as f:
            for ann in tile_annotations:
                c, cx, cy, w, h = ann
                f.write(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        tiles_created += 1

    return {
        "tiles_created": tiles_created,
        "tiles_skipped": tiles_skipped,
        "source_image": str(image_path.name),
        "source_dims": f"{img_width}x{img_height}",
        "total_tiles": len(tiles),
    }


def tile_dataset(
    input_dir: str,
    output_dir: str,
    tile_size: int = 640,
    overlap: float = 0.2,
    min_visibility: float = 0.3,
    save_empty_tiles: bool = False,
    num_workers: int = 4,
) -> dict:
    """
    Tile an entire dataset (images + labels).

    Expected input structure:
        input_dir/
        ├── images/
        └── labels/

    Output structure:
        output_dir/
        ├── images/
        └── labels/
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    in_images = input_path / "images"
    in_labels = input_path / "labels"
    out_images = output_path / "images"
    out_labels = output_path / "labels"

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(in_images.glob("*.jpg")) + list(in_images.glob("*.png"))
    )
    logger.info(f"Tiling {len(image_files)} images from {in_images}")

    total_stats = {
        "source_images": len(image_files),
        "total_tiles_created": 0,
        "total_tiles_skipped": 0,
    }

    def process_image(img_path):
        lbl_path = in_labels / (img_path.stem + ".txt")
        return tile_image_and_labels(
            img_path, lbl_path if lbl_path.exists() else None,
            out_images, out_labels,
            tile_size, overlap, min_visibility, save_empty_tiles
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_image, p): p for p in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tiling"):
            stats = future.result()
            total_stats["total_tiles_created"] += stats["tiles_created"]
            total_stats["total_tiles_skipped"] += stats["tiles_skipped"]

    logger.success(
        f"Tiling complete: {total_stats['total_tiles_created']} tiles created, "
        f"{total_stats['total_tiles_skipped']} empty tiles skipped"
    )

    # Save metadata
    with open(output_path / "tiling_metadata.json", "w") as f:
        json.dump({
            **total_stats,
            "tile_size": tile_size,
            "overlap": overlap,
            "min_visibility": min_visibility,
        }, f, indent=2)

    return total_stats


def main():
    parser = argparse.ArgumentParser(description="Tile satellite images for YOLO training")
    parser.add_argument("--input", required=True, help="Input directory with images/ and labels/")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--min-visibility", type=float, default=0.3)
    parser.add_argument("--save-empty", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tile_dataset(
        input_dir=args.input,
        output_dir=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
        save_empty_tiles=args.save_empty,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
