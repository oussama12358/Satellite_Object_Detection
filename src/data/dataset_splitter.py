"""
Dataset Splitter with Stratified Splits
=========================================
Creates reproducible train/val/test splits for tiled satellite imagery.

Design Decision:
    Naive random splits risk data leakage: tiles from the same source image
    can appear in both train and val, giving inflated metrics. We split at
    the SOURCE IMAGE level, not tile level.
"""

import json
import os
import random
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from loguru import logger
from tqdm import tqdm


def get_tile_source_image(tile_name: str) -> str:
    """Extract source image name from tile filename (e.g., P0001__tile_0042 → P0001)."""
    return tile_name.split("__tile_")[0]


def split_by_source_image(
    images_dir: Path,
    labels_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split tiles into train/val/test while keeping all tiles from the
    same source image in the same split (prevents data leakage).

    Returns:
        {"train": [...tile_stems...], "val": [...], "test": [...]}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Group tiles by source image
    all_tiles = [p.stem for p in sorted(images_dir.glob("*.jpg"))]
    source_to_tiles = defaultdict(list)
    for tile in all_tiles:
        source = get_tile_source_image(tile)
        source_to_tiles[source].append(tile)

    source_images = sorted(source_to_tiles.keys())
    random.seed(seed)
    random.shuffle(source_images)

    n = len(source_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_sources = source_images[:n_train]
    val_sources = source_images[n_train:n_train + n_val]
    test_sources = source_images[n_train + n_val:]

    splits = {
        "train": [t for s in train_sources for t in source_to_tiles[s]],
        "val": [t for s in val_sources for t in source_to_tiles[s]],
        "test": [t for s in test_sources for t in source_to_tiles[s]],
    }

    logger.info(f"Split stats (source images): train={len(train_sources)}, "
                f"val={len(val_sources)}, test={len(test_sources)}")
    logger.info(f"Split stats (tiles): train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")

    return splits


def create_split_directories(
    processed_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    hard_link: bool = True,
) -> None:
    """
    Create YOLO-compatible split directory structure.

    Output:
        output_dir/
        ├── train/images/  ├── train/labels/
        ├── val/images/    ├── val/labels/
        └── test/images/   └── test/labels/
    """
    proc_path = Path(processed_dir)
    out_path = Path(output_dir)

    images_dir = proc_path / "images"
    labels_dir = proc_path / "labels"

    splits = split_by_source_image(
        images_dir, labels_dir,
        train_ratio, val_ratio, test_ratio, seed
    )

    # Save split manifest for reproducibility
    manifest = {
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "splits": {k: sorted(v) for k, v in splits.items()},
    }
    with open(out_path / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for split_name, tile_stems in splits.items():
        split_img_dir = out_path / split_name / "images"
        split_lbl_dir = out_path / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in tqdm(tile_stems, desc=f"Creating {split_name} split"):
            src_img = images_dir / f"{stem}.jpg"
            src_lbl = labels_dir / f"{stem}.txt"

            dst_img = split_img_dir / f"{stem}.jpg"
            dst_lbl = split_lbl_dir / f"{stem}.txt"

            if src_img.exists() and not dst_img.exists():
                if hard_link:
                    try:
                        os.link(src_img, dst_img)
                    except Exception:
                        shutil.copy2(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)

            if src_lbl.exists() and not dst_lbl.exists():
                if hard_link:
                    try:
                        os.link(src_lbl, dst_lbl)
                    except Exception:
                        shutil.copy2(src_lbl, dst_lbl)
                else:
                    shutil.copy2(src_lbl, dst_lbl)

    logger.success(f"Dataset splits created at {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Split tiled dataset into train/val/test")
    parser.add_argument("--processed", required=True, help="Processed tiles directory")
    parser.add_argument("--output", required=True, help="Output splits directory")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_split_directories(
        args.processed, args.output,
        args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )


if __name__ == "__main__":
    main()
