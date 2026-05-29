"""
xView Dataset → YOLO Format Converter
========================================
Converts the xView dataset (GeoJSON annotations + GeoTIFF imagery)
to YOLO format for training.

xView format:
    - Images: GeoTIFF with WGS-84 coordinates
    - Labels: GeoJSON FeatureCollection with bounding boxes
      { "type": "FeatureCollection", "features": [
          { "type": "Feature",
            "geometry": { "type": "Point", "coordinates": [lon, lat] },
            "properties": { "bounds_imcoords": "x1,y1,x2,y2",
                            "type_id": 73, "image_id": "1.tif" } }
      ]}

xView has 60 fine-grained classes. We map these to our
aerospace-domain taxonomy (aircraft, ship, vehicle, storage-tank).
"""

import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from loguru import logger
from tqdm import tqdm


# xView class ID → display name mapping (subset shown, full list has 60 classes)
XVIEW_CLASS_NAMES = {
    11: "fixed-wing-aircraft",
    12: "small-aircraft",
    13: "cargo-plane",
    15: "helicopter",
    17: "passenger-vehicle",
    18: "small-car",
    19: "bus",
    20: "pickup-truck",
    21: "utility-truck",
    23: "truck",
    24: "cargo-truck",
    25: "truck-w-box",
    26: "truck-tractor",
    27: "trailer",
    28: "truck-w-flatbed",
    29: "truck-w-liquid",
    32: "crane-truck",
    33: "railway-vehicle",
    34: "passenger-car",
    35: "cargo/container-car",
    36: "flat-car",
    37: "tank-car",
    38: "locomotive",
    40: "maritime-vessel",
    41: "motorboat",
    42: "sailboat",
    44: "tugboat",
    45: "barge",
    47: "fishing-vessel",
    49: "ferry",
    50: "yacht",
    51: "container-ship",
    52: "oil-tanker",
    53: "engineering-vehicle",
    54: "tower-crane",
    55: "container-crane",
    56: "reach-stacker",
    57: "straddle-carrier",
    59: "mobile-crane",
    60: "dump-truck",
    61: "haul-truck",
    62: "scraper/tractor",
    63: "front-loader/bulldozer",
    64: "excavator",
    65: "cement-mixer",
    66: "ground-grader",
    71: "hut/tent",
    72: "shed",
    73: "building",
    74: "aircraft-hangar",
    76: "damaged-building",
    77: "facility",
    79: "construction-site",
    83: "vehicle-lot",
    84: "helipad",
    86: "storage-tank",
    89: "shipping-container-lot",
    91: "shipping-container",
    93: "pylon",
    94: "tower",
}

# Aerospace taxonomy mapping: xView type_id → our class name
XVIEW_TO_AEROSPACE = {
    # Aircraft
    11: "aircraft", 12: "aircraft", 13: "aircraft", 15: "aircraft",
    # Ships / maritime
    40: "ship", 41: "ship", 42: "ship", 44: "ship", 45: "ship",
    47: "ship", 49: "ship", 50: "ship", 51: "ship", 52: "ship",
    # Ground vehicles
    17: "vehicle", 18: "vehicle", 19: "vehicle", 20: "vehicle",
    21: "vehicle", 23: "vehicle", 24: "vehicle", 25: "vehicle",
    32: "vehicle", 53: "vehicle", 60: "vehicle", 61: "vehicle",
    63: "vehicle", 64: "vehicle",
    # Storage tanks
    86: "storage-tank",
}

AEROSPACE_CLASSES = ["aircraft", "ship", "vehicle", "storage-tank"]


def parse_xview_geojson(
    geojson_path: str,
    class_filter: Optional[List[str]] = None,
    use_aerospace_mapping: bool = True,
) -> Dict[str, List[Tuple]]:
    """
    Parse xView GeoJSON annotation file.

    Args:
        geojson_path: Path to xView labels.json
        class_filter: List of class names to keep (None = keep all)
        use_aerospace_mapping: Map 60 xView classes to aerospace taxonomy

    Returns:
        Dict mapping image_id → list of (class_name, x1, y1, x2, y2)
    """
    with open(geojson_path) as f:
        data = json.load(f)

    class_list = AEROSPACE_CLASSES if use_aerospace_mapping else list(set(XVIEW_CLASS_NAMES.values()))
    if class_filter:
        class_list = [c for c in class_list if c in class_filter]

    image_annotations = defaultdict(list)

    for feature in data.get("features", []):
        props = feature.get("properties", {})
        type_id = props.get("type_id")
        image_id = props.get("image_id", "")
        bounds_str = props.get("bounds_imcoords", "")

        if not bounds_str or type_id is None:
            continue

        # Map to class name
        if use_aerospace_mapping:
            class_name = XVIEW_TO_AEROSPACE.get(type_id)
        else:
            class_name = XVIEW_CLASS_NAMES.get(type_id)

        if class_name is None or (class_filter and class_name not in class_filter):
            continue

        # Parse bounding box
        try:
            x1, y1, x2, y2 = map(float, bounds_str.split(","))
        except (ValueError, AttributeError):
            continue

        # Validate box
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 2 or (y2 - y1) < 2:
            continue

        image_annotations[image_id].append((class_name, x1, y1, x2, y2))

    logger.info(
        f"Parsed {sum(len(v) for v in image_annotations.values())} annotations "
        f"across {len(image_annotations)} images"
    )
    return dict(image_annotations)


def convert_xview_dataset(
    images_dir: str,
    geojson_path: str,
    output_dir: str,
    use_aerospace_mapping: bool = True,
    class_filter: Optional[List[str]] = None,
    copy_images: bool = True,
) -> dict:
    """
    Convert xView dataset to YOLO format.

    Args:
        images_dir: Directory containing xView .tif images
        geojson_path: Path to xView labels GeoJSON
        output_dir: Output directory
        use_aerospace_mapping: Collapse 60 classes to 4 aerospace classes
        class_filter: Only include specific classes
        copy_images: Copy images to output directory

    Returns:
        Conversion statistics
    """
    images_path = Path(images_dir)
    output_path = Path(output_dir)

    out_images = output_path / "images"
    out_labels = output_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # Determine class list
    if use_aerospace_mapping:
        class_list = AEROSPACE_CLASSES
    else:
        class_list = sorted(set(XVIEW_CLASS_NAMES.values()))

    # Parse GeoJSON
    annotations = parse_xview_geojson(
        geojson_path, class_filter=class_filter,
        use_aerospace_mapping=use_aerospace_mapping
    )

    stats = {
        "total_images": 0,
        "converted": 0,
        "skipped": 0,
        "total_annotations": 0,
        "class_counts": {c: 0 for c in class_list},
    }

    for image_id, boxes in tqdm(annotations.items(), desc="Converting xView"):
        # Support both .tif and .png extensions
        img_path = None
        for ext in [".tif", ".tiff", ".png", ".jpg"]:
            candidate = images_path / (Path(image_id).stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            stats["skipped"] += 1
            continue

        # Get image dimensions
        try:
            from PIL import Image as PILImage
            with PILImage.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            logger.warning(f"Cannot read {img_path}: {e}")
            stats["skipped"] += 1
            continue

        stats["total_images"] += 1

        # Write YOLO labels
        yolo_lines = []
        for class_name, x1, y1, x2, y2 in boxes:
            if class_name not in class_list:
                continue
            class_id = class_list.index(class_name)

            # Clamp to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            if w < 0.001 or h < 0.001:
                continue

            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats["class_counts"][class_name] += 1
            stats["total_annotations"] += 1

        if not yolo_lines:
            stats["skipped"] += 1
            continue

        # Save label
        label_out = out_labels / (img_path.stem + ".txt")
        with open(label_out, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        # Copy/link image
        img_out = out_images / img_path.name
        if copy_images and not img_out.exists():
            shutil.copy2(img_path, img_out)

        stats["converted"] += 1

    # Save class reference
    with open(output_path / "classes.txt", "w") as f:
        for i, cls in enumerate(class_list):
            f.write(f"{i}: {cls}\n")

    with open(output_path / "conversion_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.success(
        f"xView conversion: {stats['converted']} images, "
        f"{stats['total_annotations']} annotations"
    )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert xView to YOLO format")
    parser.add_argument("--images", required=True, help="xView images directory")
    parser.add_argument("--geojson", required=True, help="xView labels GeoJSON path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--no-aerospace-mapping", action="store_true",
                        help="Keep all 60 xView classes instead of mapping to aerospace taxonomy")
    parser.add_argument("--classes", nargs="+", help="Filter to specific class names")
    args = parser.parse_args()

    convert_xview_dataset(
        images_dir=args.images,
        geojson_path=args.geojson,
        output_dir=args.output,
        use_aerospace_mapping=not args.no_aerospace_mapping,
        class_filter=args.classes,
    )


if __name__ == "__main__":
    main()
