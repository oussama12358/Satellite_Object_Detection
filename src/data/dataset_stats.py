"""
Dataset Statistics & EDA
==========================
Analyzes class distributions, object size statistics, and data quality
for the satellite detection dataset.

Produces:
    - Class frequency bar chart
    - Object size distribution (small/medium/large breakdown)
    - Aspect ratio analysis
    - Per-split statistics table
    - Imbalance ratio and suggested class weights
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from loguru import logger


# COCO-standard size buckets (normalized area)
SIZE_BUCKETS = {
    "tiny":   (0,       0.001),   # < 32×32 in 640px image
    "small":  (0.001,   0.01),    # 32-100px equivalent
    "medium": (0.01,    0.1),
    "large":  (0.1,     1.0),
}


def analyze_label_file(
    label_path: Path,
    class_names: Dict[int, str],
) -> List[Dict]:
    """Parse a single YOLO label file and return annotation metadata."""
    annotations = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            area = w * h
            aspect = w / h if h > 0 else 0
            annotations.append({
                "class_id": cls_id,
                "class_name": class_names.get(cls_id, f"class_{cls_id}"),
                "cx": cx, "cy": cy,
                "w": w, "h": h,
                "area": area,
                "aspect_ratio": aspect,
            })
    return annotations


def classify_size(area: float) -> str:
    for size_name, (lo, hi) in SIZE_BUCKETS.items():
        if lo <= area < hi:
            return size_name
    return "large"


def compute_dataset_stats(
    splits_dir: str,
    class_names_path: Optional[str] = None,
    output_dir: str = "data/stats",
) -> Dict:
    """
    Compute comprehensive statistics for all splits.

    Args:
        splits_dir: Root directory with train/val/test splits
        class_names_path: Path to classes.txt (optional)
        output_dir: Where to save charts and JSON report

    Returns:
        Stats dict
    """
    splits_path = Path(splits_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load class names
    class_names = {}
    if class_names_path and Path(class_names_path).exists():
        with open(class_names_path) as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) == 2:
                    class_names[int(parts[0])] = parts[1].strip()

    all_stats = {}

    for split in ["train", "val", "test"]:
        lbl_dir = splits_path / split / "labels"
        img_dir = splits_path / split / "images"

        if not lbl_dir.exists():
            continue

        label_files = list(lbl_dir.glob("*.txt"))
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

        all_annotations = []
        for lf in label_files:
            all_annotations.extend(analyze_label_file(lf, class_names))

        class_counts = Counter(a["class_name"] for a in all_annotations)
        size_counts = Counter(classify_size(a["area"]) for a in all_annotations)
        areas = [a["area"] for a in all_annotations]
        aspects = [a["aspect_ratio"] for a in all_annotations]

        per_class_sizes = defaultdict(lambda: Counter())
        for a in all_annotations:
            per_class_sizes[a["class_name"]][classify_size(a["area"])] += 1

        split_stats = {
            "num_images": len(image_files),
            "num_label_files": len(label_files),
            "total_annotations": len(all_annotations),
            "avg_annotations_per_image": (
                len(all_annotations) / max(len(label_files), 1)
            ),
            "class_counts": dict(class_counts),
            "size_distribution": dict(size_counts),
            "area_stats": {
                "mean": float(np.mean(areas)) if areas else 0,
                "median": float(np.median(areas)) if areas else 0,
                "min": float(np.min(areas)) if areas else 0,
                "max": float(np.max(areas)) if areas else 0,
                "p25": float(np.percentile(areas, 25)) if areas else 0,
                "p75": float(np.percentile(areas, 75)) if areas else 0,
            },
            "aspect_ratio_stats": {
                "mean": float(np.mean(aspects)) if aspects else 0,
                "median": float(np.median(aspects)) if aspects else 0,
            },
        }
        all_stats[split] = split_stats

    # Compute class imbalance weights (for training config)
    if "train" in all_stats:
        train_counts = all_stats["train"]["class_counts"]
        max_count = max(train_counts.values()) if train_counts else 1
        class_weights = {
            cls: round(max_count / count, 3)
            for cls, count in train_counts.items()
        }
        all_stats["suggested_class_weights"] = class_weights
        all_stats["imbalance_ratio"] = (
            max(train_counts.values()) / max(min(train_counts.values()), 1)
            if train_counts else 1
        )

    # Save JSON report
    report_path = output_path / "dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Stats report saved: {report_path}")

    # Generate charts
    _plot_class_distribution(all_stats, output_path)
    _plot_size_distribution(all_stats, output_path)
    _print_summary_table(all_stats)

    return all_stats


def _plot_class_distribution(stats: Dict, output_path: Path) -> None:
    """Bar chart of per-class annotation counts across splits."""
    splits = [s for s in ["train", "val", "test"] if s in stats]
    if not splits:
        return

    # Collect all class names
    all_classes = sorted(set(
        cls for s in splits
        for cls in stats[s].get("class_counts", {}).keys()
    ))
    if not all_classes:
        return

    x = np.arange(len(all_classes))
    width = 0.25
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(max(12, len(all_classes) * 1.2), 6))

    for i, (split, color) in enumerate(zip(splits, colors)):
        counts = [stats[split].get("class_counts", {}).get(cls, 0) for cls in all_classes]
        ax.bar(x + i * width, counts, width, label=split.capitalize(), color=color, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(all_classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Annotation Count", fontsize=12)
    ax.set_title("Class Distribution Across Splits", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")  # Log scale handles extreme imbalances

    plt.tight_layout()
    plt.savefig(output_path / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_size_distribution(stats: Dict, output_path: Path) -> None:
    """Pie charts showing object size distribution per split."""
    splits = [s for s in ["train", "val", "test"] if s in stats and stats[s].get("size_distribution")]
    if not splits:
        return

    fig, axes = plt.subplots(1, len(splits), figsize=(5 * len(splits), 5))
    if len(splits) == 1:
        axes = [axes]

    size_order = ["tiny", "small", "medium", "large"]
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    for ax, split in zip(axes, splits):
        size_dist = stats[split]["size_distribution"]
        sizes = [size_dist.get(s, 0) for s in size_order]
        labels_text = [f"{s}\n({v:,})" for s, v in zip(size_order, sizes)]
        non_zero = [
            (size_count, label_text, color)
            for size_count, label_text, color in zip(sizes, labels_text, colors)
            if size_count > 0
        ]
        if non_zero:
            sz, lb, cl = zip(*non_zero)
            ax.pie(sz, labels=lb, colors=cl, autopct="%1.1f%%",
                   startangle=90, textprops={"fontsize": 9})
        ax.set_title(f"{split.capitalize()} Split\n({stats[split]['total_annotations']:,} objects)",
                    fontsize=11, fontweight="bold")

    plt.suptitle("Object Size Distribution (by Normalized Area)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "size_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def _print_summary_table(stats: Dict) -> None:
    print("\n" + "="*70)
    print("  Dataset Statistics Summary")
    print("="*70)
    print(f"  {'Split':<10} {'Images':>8} {'Annotations':>14} {'Avg/Img':>10}")
    print("-"*70)
    for split in ["train", "val", "test"]:
        if split not in stats:
            continue
        s = stats[split]
        print(f"  {split:<10} {s['num_images']:>8,} {s['total_annotations']:>14,} "
              f"{s['avg_annotations_per_image']:>10.1f}")
    print("-"*70)
    if "imbalance_ratio" in stats:
        print(f"  Class imbalance ratio: {stats['imbalance_ratio']:.1f}×")
    if "suggested_class_weights" in stats:
        print("\n  Suggested class weights for training:")
        for cls, w in sorted(stats["suggested_class_weights"].items(), key=lambda x: -x[1]):
            bar = "█" * min(int(w * 5), 30)
            print(f"    {cls:<25} {w:>6.3f}  {bar}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--splits", required=True, help="Splits directory")
    parser.add_argument("--classes", help="Path to classes.txt")
    parser.add_argument("--output", default="data/stats")
    args = parser.parse_args()

    compute_dataset_stats(args.splits, args.classes, args.output)


if __name__ == "__main__":
    main()
