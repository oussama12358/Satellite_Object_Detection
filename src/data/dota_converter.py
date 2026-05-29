"""
DOTA → YOLO Annotation Converter
==================================
Converts DOTA v1.0/v1.5/v2.0 polygon annotations to YOLO horizontal bounding box format.

DOTA annotation format:
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

YOLO format:
    class_id cx cy w h  (normalized 0-1)

Design Decision:
    DOTA uses oriented bounding boxes (OBB). We convert to axis-aligned HBB by computing
    the minimum enclosing rectangle of the polygon. This is a trade-off: we lose orientation
    information but gain compatibility with standard YOLOv8 detection head.
    For production aerospace systems, consider YOLOv8-OBB for full OBB support.
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
from loguru import logger


# ── DOTA class taxonomy ────────────────────────────────────────────────────────
DOTA_V15_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "large-vehicle", "small-vehicle",
    "helicopter", "roundabout", "soccer-ball-field",
    "swimming-pool", "container-crane"
]

DOTA_V10_CLASSES = [
    "plane", "ship", "storage-tank", "baseball-diamond",
    "tennis-court", "basketball-court", "ground-track-field",
    "harbor", "bridge", "large-vehicle", "small-vehicle",
    "helicopter", "roundabout", "soccer-ball-field", "swimming-pool"
]

# Aerospace-focused subset mapping
AEROSPACE_CLASS_MAP = {
    "plane": "aircraft",
    "helicopter": "aircraft",
    "ship": "ship",
    "large-vehicle": "vehicle",
    "small-vehicle": "vehicle",
    "storage-tank": "storage-tank",
}


def polygon_to_hbb(polygon_pts: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert polygon (8 coords) to axis-aligned bounding box.

    Args:
        polygon_pts: [x1, y1, x2, y2, x3, y3, x4, y4]

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    coords = np.array(polygon_pts, dtype=np.float32).reshape(4, 2)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return float(x_min), float(y_min), float(x_max), float(y_max)


def hbb_to_yolo(
    x_min: float, y_min: float, x_max: float, y_max: float,
    img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute HBB to normalized YOLO format.

    Returns:
        (cx, cy, w, h) all in [0, 1]
    """
    cx = (x_min + x_max) / 2.0 / img_width
    cy = (y_min + y_max) / 2.0 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height

    # Clip to valid range
    cx = np.clip(cx, 0, 1)
    cy = np.clip(cy, 0, 1)
    w = np.clip(w, 0, 1)
    h = np.clip(h, 0, 1)

    return cx, cy, w, h


def parse_dota_label_file(
    label_path: Path,
    class_list: List[str],
    img_width: int,
    img_height: int,
    skip_difficult: bool = True,
    class_filter: Optional[List[str]] = None,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a single DOTA annotation file.

    Args:
        label_path: Path to .txt annotation file
        class_list: Ordered list of class names (defines class IDs)
        img_width: Image width in pixels
        img_height: Image height in pixels
        skip_difficult: Skip annotations marked as difficult
        class_filter: If set, only include these class names

    Returns:
        List of (class_id, cx, cy, w, h) tuples
    """
    annotations = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("imagesource") or line.startswith("gsd"):
            continue

        parts = line.split()
        if len(parts) < 9:
            logger.warning(f"Malformed line in {label_path}: {line}")
            continue

        coords = list(map(float, parts[:8]))
        category = parts[8].lower()
        difficulty = int(parts[9]) if len(parts) > 9 else 0

        # Skip difficult instances (ambiguous in satellite imagery)
        if skip_difficult and difficulty == 1:
            continue

        # Filter by requested classes
        if class_filter and category not in class_filter:
            continue

        if category not in class_list:
            continue

        class_id = class_list.index(category)
        x_min, y_min, x_max, y_max = polygon_to_hbb(coords)

        # Sanity check: skip degenerate boxes
        if (x_max - x_min) < 2 or (y_max - y_min) < 2:
            continue

        cx, cy, w, h = hbb_to_yolo(x_min, y_min, x_max, y_max, img_width, img_height)
        annotations.append((class_id, cx, cy, w, h))

    return annotations


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) without loading full image using PIL."""
    from PIL import Image
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def convert_dota_dataset(
    input_dir: str,
    output_dir: str,
    version: str = "v1.0",
    skip_difficult: bool = True,
    class_filter: Optional[List[str]] = None,
    copy_images: bool = True,
) -> Dict:
    """
    Convert full DOTA dataset to YOLO format.

    Expected input structure:
        input_dir/
        ├── images/         # .png or .jpg files
        └── labelTxt/       # .txt annotation files

    Output structure:
        output_dir/
        ├── images/
        └── labels/

    Args:
        input_dir: DOTA dataset root
        output_dir: Processed output root
        version: "v1.0" or "v1.5" or "v2.0"
        skip_difficult: Ignore difficult=1 instances
        class_filter: Subset of classes to keep
        copy_images: Copy images to output dir (set False to symlink)

    Returns:
        Conversion statistics dict
    """
    version_map = {"v1.0": DOTA_V10_CLASSES, "v1.5": DOTA_V15_CLASSES}
    class_list = version_map.get(version, DOTA_V10_CLASSES)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    images_in = input_path / "images"
    labels_in = input_path / "labelTxt"
    images_out = output_path / "images"
    labels_out = output_path / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(images_in.glob("*.png")) + list(images_in.glob("*.jpg")))
    logger.info(f"Found {len(image_files)} images in {images_in}")

    stats = {
        "total_images": len(image_files),
        "converted": 0,
        "skipped_no_label": 0,
        "skipped_no_annotations": 0,
        "total_annotations": 0,
        "class_counts": {cls: 0 for cls in class_list},
    }

    for img_path in tqdm(image_files, desc="Converting DOTA annotations"):
        label_path = labels_in / (img_path.stem + ".txt")

        if not label_path.exists():
            stats["skipped_no_label"] += 1
            continue

        try:
            img_width, img_height = get_image_dimensions(img_path)
        except Exception as e:
            logger.warning(f"Cannot read image {img_path}: {e}")
            continue

        annotations = parse_dota_label_file(
            label_path, class_list, img_width, img_height,
            skip_difficult=skip_difficult,
            class_filter=class_filter,
        )

        if not annotations:
            stats["skipped_no_annotations"] += 1
            continue

        # Write YOLO label
        yolo_label_path = labels_out / (img_path.stem + ".txt")
        with open(yolo_label_path, "w") as f:
            for class_id, cx, cy, w, h in annotations:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                stats["class_counts"][class_list[class_id]] += 1
                stats["total_annotations"] += 1

        # Copy/link image
        dest_img = images_out / img_path.name
        if copy_images:
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)
        else:
            if not dest_img.exists():
                os.symlink(img_path.resolve(), dest_img)

        stats["converted"] += 1

    # Save class list for reference
    with open(output_path / "classes.txt", "w") as f:
        for i, cls in enumerate(class_list):
            f.write(f"{i}: {cls}\n")

    # Save conversion report
    with open(output_path / "conversion_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.success(
        f"Conversion complete: {stats['converted']}/{stats['total_images']} images, "
        f"{stats['total_annotations']} annotations"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert DOTA dataset to YOLO format")
    parser.add_argument("--input", required=True, help="DOTA dataset root directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--version", default="v1.0", choices=["v1.0", "v1.5", "v2.0"])
    parser.add_argument("--skip-difficult", action="store_true", default=True)
    parser.add_argument("--classes", nargs="+", help="Filter to specific classes")
    parser.add_argument("--no-copy-images", action="store_true")
    args = parser.parse_args()

    stats = convert_dota_dataset(
        input_dir=args.input,
        output_dir=args.output,
        version=args.version,
        skip_difficult=args.skip_difficult,
        class_filter=args.classes,
        copy_images=not args.no_copy_images,
    )

    print("\n📊 Conversion Statistics:")
    print(f"  Images converted:  {stats['converted']}")
    print(f"  Total annotations: {stats['total_annotations']}")
    print("\n  Per-class counts:")
    for cls, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cls:25s}: {count:6d}")


if __name__ == "__main__":
    main()
