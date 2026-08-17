import argparse
import os
import random
import shutil
import sys
import yaml
from pathlib import Path
from collections import defaultdict

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from config import (
    ROBOFLOW_API_KEY,
    ROBOFLOW_DATASETS,
    TARGET_CLASSES,
    NUM_CLASSES,
    LABEL_ALIASES,
    RAW_DIR,
    MERGED_DIR,
    DATA_DIR,
    DATA_YAML,
    CLASS_MAX_ANNOTATIONS,
    CLASS_OVERSAMPLE_MULTIPLIER,
)

def ensure_dirs():
    """Create all necessary data directories."""
    for d in [RAW_DIR, MERGED_DIR,
              MERGED_DIR / "train" / "images",
              MERGED_DIR / "train" / "labels",
              MERGED_DIR / "val" / "images",
              MERGED_DIR / "val" / "labels"]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"Directory structure created under {DATA_DIR}")

def download_datasets(dry_run=False):
    """
    Download all configured Roboflow Universe datasets.
    Each dataset is saved to data/raw/<workspace>_<project>/
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("roboflow package not found. Installing...")
        os.system(f"{sys.executable} -m pip install roboflow")
        from roboflow import Roboflow

    if not ROBOFLOW_API_KEY:
        print("ERROR: No Roboflow API key found!")
        print("    Set ROBOFLOW_API_KEY env variable or update config.py")
        sys.exit(1)

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    downloaded = []

    for workspace, project_name, version, description in ROBOFLOW_DATASETS:
        dataset_id = f"{workspace}/{project_name}/v{version}"
        print(f"\n{'─' * 60}")
        print(f"  Dataset: {dataset_id}")
        print(f"  Description: {description}")

        if dry_run:
            print(f"  [DRY RUN] Skipping download")
            continue

        download_path = RAW_DIR / f"{workspace}_{project_name}"

        # Skip if already downloaded and valid (contains both YAML and images)
        yaml_exists = any(download_path.rglob("data.yaml")) or any(download_path.glob("*.yaml"))
        has_images = any(download_path.rglob("*.jpg")) or any(download_path.rglob("*.png")) or any(download_path.rglob("*.jpeg"))
        
        if download_path.exists() and yaml_exists and has_images:
            print(f"  [SKIP] Already exists and verified at {download_path}")
            downloaded.append(download_path)
            continue
        elif download_path.exists():
            print(f"  Found existing directory {download_path} but it is incomplete (missing YAML or images).")
            print(f"      Cleaning and redownloading...")
            shutil.rmtree(str(download_path))

        try:
            project = rf.workspace(workspace).project(project_name)
            ds = project.version(version)
            ds.download("yolov8", location=str(download_path))
            print(f"  Downloaded to {download_path}")
            downloaded.append(download_path)
        except Exception as e:
            print(f"  ERROR: Failed to download: {e}")
            print(f"      This dataset may not be publicly accessible or the")
            print(f"      workspace/project name may have changed on Roboflow.")
            print(f"      Skipping this dataset and continuing...")

    print(f"\n{'═' * 60}")
    print(f"  Downloaded {len(downloaded)} / {len(ROBOFLOW_DATASETS)} datasets")
    return downloaded

def parse_dataset_yaml(dataset_path):
    """
    Read the data.yaml from a downloaded Roboflow dataset and return
    a mapping of class_index -> class_name.
    """
    yaml_candidates = list(dataset_path.rglob("data.yaml"))
    if not yaml_candidates:
        # Try the root-level YAML
        yaml_candidates = list(dataset_path.glob("*.yaml"))

    if not yaml_candidates:
        print(f"  No YAML config found in {dataset_path}")
        return None

    yaml_path = yaml_candidates[0]
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", {})

    # Handle both dict format {0: "cow", 1: "dog"} and list format ["cow", "dog"]
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    else:
        print(f"  Unexpected 'names' format in {yaml_path}: {type(names)}")
        return None

def build_label_remap(source_classes):
    """
    Given a dict of {idx: class_name} from a source dataset, build a remap
    dict: {source_idx: target_idx} where target_idx is from TARGET_CLASSES.
    Returns None for classes that don't map to any target.
    """
    target_lookup = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    remap = {}

    for src_idx, src_name in source_classes.items():
        # Try exact match first
        unified = LABEL_ALIASES.get(src_name)
        if unified is None:
            # Try lowercase match
            unified = LABEL_ALIASES.get(src_name.lower())
        if unified is None:
            # Try with underscores replaced by spaces
            unified = LABEL_ALIASES.get(src_name.replace("_", " "))
        if unified is None:
            # Try stripping whitespace
            unified = LABEL_ALIASES.get(src_name.strip())

        if unified and unified in target_lookup:
            remap[src_idx] = target_lookup[unified]
        else:
            remap[src_idx] = None  # This class is not one of our targets

    return remap

def remap_and_copy_split(dataset_path, split_name, remap, stats, file_counter, train_img_count=None):
    """
    Process a single split (train/valid/test) from a downloaded dataset:
    - Copy images to merged directory
    - Remap labels and copy to merged directory
    - Apply class balancing (undersampling/oversampling) to train split
    - Track statistics

    Returns the updated file counter.
    """
    # Roboflow uses "valid" but our merged structure uses "val"
    target_split = "val" if split_name in ("valid", "test") else "train"

    images_dir = dataset_path / split_name / "images"
    labels_dir = dataset_path / split_name / "labels"

    if not images_dir.exists():
        return file_counter

    target_images = MERGED_DIR / target_split / "images"
    target_labels = MERGED_DIR / target_split / "labels"

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        # Corresponding label file
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Read and remap labels
        new_lines = []
        has_valid_labels = False
        apply_balancing = (target_split == "train")
        img_classes = set()
        if train_img_count is None:
            train_img_count = defaultdict(int)

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # Malformed line

                src_class_idx = int(parts[0])
                new_class_idx = remap.get(src_class_idx)

                if new_class_idx is not None:
                    class_name = TARGET_CLASSES[new_class_idx]

                    # 1. IMAGE-LEVEL undersampling (cap per unique image, not per annotation).
                    # Old bug: counted annotation lines → goat (5/image) got ~700 images at cap=3500.
                    # New fix: count images → every class gets up to CLASS_MAX_ANNOTATIONS images.
                    if apply_balancing and class_name in CLASS_MAX_ANNOTATIONS:
                        if train_img_count[class_name] >= CLASS_MAX_ANNOTATIONS[class_name]:
                            continue  # This class is full — drop its boxes but keep the image

                    parts[0] = str(new_class_idx)
                    new_lines.append(" ".join(parts))
                    has_valid_labels = True
                    img_classes.add(class_name)

        # Only copy if we have at least one valid label
        if not has_valid_labels:
            continue

        # Update per-class image count (before oversampling copies)
        if apply_balancing:
            for cname in img_classes:
                train_img_count[cname] += 1

        # 2. Oversampling minority classes
        copies = 1
        if apply_balancing:
            max_mult = 1.0
            for cname in img_classes:
                if cname in CLASS_OVERSAMPLE_MULTIPLIER:
                    max_mult = max(max_mult, CLASS_OVERSAMPLE_MULTIPLIER[cname])

            base_copies = int(max_mult)
            fractional_part = max_mult - base_copies
            copies = base_copies + (1 if random.random() < fractional_part else 0)

        # Copy the image and write labels copies times
        for _ in range(copies):
            file_counter += 1
            new_stem = f"agd_{file_counter:06d}"
            new_img_name = new_stem + img_path.suffix.lower()
            new_lbl_name = new_stem + ".txt"

            try:
                shutil.copy2(str(img_path), str(target_images / new_img_name))
            except Exception:
                try:
                    shutil.copy(str(img_path), str(target_images / new_img_name))
                except Exception as e:
                    print(f"Warning copying {img_path.name}: {e}")
                    continue

            with open(target_labels / new_lbl_name, "w") as f:
                f.write("\n".join(new_lines) + "\n")

            for parts_line in new_lines:
                class_idx = int(parts_line.split()[0])
                class_name = TARGET_CLASSES[class_idx]
                stats[target_split][class_name] += 1

    return file_counter

def merge_datasets(downloaded_paths):
    """
    Merge all downloaded datasets into a single unified YOLO dataset.
    Handles label remapping, deduplication, and train/val organization.
    """
    print(f"\n{'═' * 60}")
    print("  MERGING DATASETS")
    print(f"{'═' * 60}")

    # Track per-class statistics
    stats = {
        "train": defaultdict(int),
        "val":   defaultdict(int),
    }
    # Track IMAGES per class (not annotations) for balancing.
    # CLASS_MAX_ANNOTATIONS config is misleadingly named — we now use it as image cap.
    train_img_count = defaultdict(int)  # images seen per class in training split
    file_counter = 0
    datasets_used = 0

    for dataset_path in downloaded_paths:
        print(f"\n  Processing: {dataset_path.name}")

        # Parse source dataset's class definitions
        source_classes = parse_dataset_yaml(dataset_path)
        if source_classes is None:
            print(f"  [SKIP] Could not parse class definitions")
            continue

        # Build remap from source classes to our target classes
        remap = build_label_remap(source_classes)

        # Count how many classes actually map to our targets
        mapped = {k: v for k, v in remap.items() if v is not None}
        if not mapped:
            print(f"  [SKIP] No classes match our target labels")
            print(f"         Source classes: {list(source_classes.values())}")
            continue

        mapped_names = [source_classes[k] for k in mapped.keys()]
        print(f"  Mapped classes: {mapped_names}")

        # Process each split
        for split_name in ["train", "valid", "test"]:
            file_counter = remap_and_copy_split(
                dataset_path, split_name, remap, stats, file_counter, train_img_count
            )

        datasets_used += 1

    print(f"\n  Merged {datasets_used} datasets ({file_counter} total images)")
    return stats, file_counter

def rebalance_val_split(val_min_annotations=300, move_fraction=0.15):
    """
    Fix classes that ended up with too few (or zero) val images after merging.

    For any class with fewer than `val_min_annotations` annotations in val:
      - Finds all training images whose label file contains ONLY that class
        (single-class images, so moving them won't hurt multi-class train images).
      - Randomly selects `move_fraction` (15%) of those images.
      - Moves image + label file from train/ → val/.

    This fixes:
      - wild_boar: 0 val annotations → moves ~500 train images to val
      - dog: 183 val annotations (too few) → tops up to ~500
    Val set is completely isolated from augmentation (augment_weak_classes.py
    and augment_nightvision.py only touch train images).
    """
    print(f"\n{'\u2550' * 60}")
    print("  REBALANCING VAL SPLIT")
    print(f"{'\u2550' * 60}")

    train_imgs = MERGED_DIR / "train" / "images"
    train_lbls = MERGED_DIR / "train" / "labels"
    val_imgs   = MERGED_DIR / "val"   / "images"
    val_lbls   = MERGED_DIR / "val"   / "labels"

    # Count current val annotations per class
    val_counts = defaultdict(int)
    for lf in val_lbls.glob("*.txt"):
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    idx = int(parts[0])
                    if 0 <= idx < NUM_CLASSES:
                        val_counts[TARGET_CLASSES[idx]] += 1

    moved_total = 0
    for cls_name in TARGET_CLASSES:
        current_val = val_counts.get(cls_name, 0)
        if current_val >= val_min_annotations:
            print(f"  [{cls_name:<12}] val={current_val:>5}  ✔ OK")
            continue

        cls_idx = TARGET_CLASSES.index(cls_name)
        print(f"  [{cls_name:<12}] val={current_val:>5}  ⚠ Below {val_min_annotations} — rebalancing...")

        # Find train images that contain ONLY this class (single-class images)
        candidates = []
        ext_list = (".jpg", ".jpeg", ".png", ".bmp")
        for lf in train_lbls.glob("*.txt"):
            with open(lf) as f:
                lines = [l.strip() for l in f if l.strip()]
            class_ids = set()
            valid = True
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    valid = False
                    break
                class_ids.add(int(parts[0]))
            if not valid or class_ids != {cls_idx}:
                continue
            # Find matching image
            img_p = None
            for ext in ext_list:
                candidate = train_imgs / (lf.stem + ext)
                if candidate.exists():
                    img_p = candidate
                    break
            if img_p:
                candidates.append((img_p, lf))

        if not candidates:
            print(f"    No single-class train images found for {cls_name}")
            # Fallback: use ANY train images containing this class
            for lf in train_lbls.glob("*.txt"):
                with open(lf) as f:
                    ids_in_file = set()
                    for ln in f:
                        parts = ln.strip().split()
                        if len(parts) >= 5:
                            ids_in_file.add(int(parts[0]))
                if cls_idx not in ids_in_file:
                    continue
                for ext in ext_list:
                    candidate = train_imgs / (lf.stem + ext)
                    if candidate.exists():
                        candidates.append((candidate, lf))
                        break

        if not candidates:
            print(f"    No train images at all for {cls_name} — skipping")
            continue

        n_move = max(1, int(len(candidates) * move_fraction))
        random.shuffle(candidates)
        to_move = candidates[:n_move]

        for img_p, lf in to_move:
            dst_img = val_imgs / img_p.name
            dst_lbl = val_lbls / lf.name
            shutil.move(str(img_p), str(dst_img))
            shutil.move(str(lf),    str(dst_lbl))
            moved_total += 1

        print(f"    Moved {n_move} images from train → val  (had {len(candidates)} candidates)")

    print(f"\n  Total images moved to val: {moved_total}")
    print(f"{'\u2550' * 60}\n")
    return moved_total

def generate_data_yaml():
    """Generate the YOLO data.yaml configuration file."""
    data_config = {
        "path": str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": NUM_CLASSES,
        "names": TARGET_CLASSES,
    }

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {DATA_YAML}")

def print_statistics(stats, total_images):
    """Display dataset statistics in a formatted table."""
    print(f"\n{'═' * 60}")
    print("  DATASET STATISTICS")
    print(f"{'═' * 60}")
    print(f"  {'Class':<15} {'Train':>8} {'Val':>8} {'Total':>8}")
    print(f"  {'─' * 15} {'─' * 8} {'─' * 8} {'─' * 8}")

    grand_train = 0
    grand_val = 0

    for cls in TARGET_CLASSES:
        train_count = stats["train"].get(cls, 0)
        val_count = stats["val"].get(cls, 0)
        total = train_count + val_count
        grand_train += train_count
        grand_val += val_count
        # Flag classes with very few samples
        flag = " ⚠️" if total < 50 else ""
        print(f"  {cls:<15} {train_count:>8} {val_count:>8} {total:>8}{flag}")

    print(f"  {'─' * 15} {'─' * 8} {'─' * 8} {'─' * 8}")
    print(f"  {'TOTAL':<15} {grand_train:>8} {grand_val:>8} {grand_train + grand_val:>8}")
    print(f"\n  Total image files: {total_images}")

    # Warn about underrepresented classes
    for cls in TARGET_CLASSES:
        total = stats["train"].get(cls, 0) + stats["val"].get(cls, 0)
        if total == 0:
            print(f"\n  [⚠ WARNING] Class '{cls}' has ZERO samples!")
            print(f"              Consider adding more datasets for this class.")
        elif total < 50:
            print(f"\n  [⚠ WARNING] Class '{cls}' has only {total} samples.")
            print(f"              This may lead to poor detection accuracy.")

def compute_stats_only():
    """Compute and display statistics from the already-merged dataset."""
    stats = {"train": defaultdict(int), "val": defaultdict(int)}
    total_images = 0

    for split in ["train", "val"]:
        labels_dir = MERGED_DIR / split / "labels"
        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            total_images += 1
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < NUM_CLASSES:
                            stats[split][TARGET_CLASSES[class_idx]] += 1

    print_statistics(stats, total_images)

def main():
    random.seed(42)
    parser = argparse.ArgumentParser(
        description="Animal Guard — Collect and merge animal detection datasets"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all datasets without downloading them"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show statistics of the already-merged dataset"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Download only, skip the merge step"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete merged directory before re-merging"
    )
    parser.add_argument(
        "--clean-raw",
        action="store_true",
        help="Delete raw download directory before re-downloading"
    )
    args = parser.parse_args()

    # ── Stats only mode ──
    if args.stats_only:
        if not MERGED_DIR.exists():
            print("ERROR: Merged dataset not found. Run collect_data.py first.")
            sys.exit(1)
        compute_stats_only()
        return

    # ── Clean mode ──
    if args.clean and MERGED_DIR.exists():
        print(f"Cleaning merged directory: {MERGED_DIR}")
        shutil.rmtree(MERGED_DIR)

    if args.clean_raw and RAW_DIR.exists():
        print(f"Cleaning raw directory: {RAW_DIR}")
        shutil.rmtree(RAW_DIR)ensure_dirs()

    # ── Download ──
    print(f"\n{'═' * 60}")
    print("  DOWNLOADING DATASETS FROM ROBOFLOW UNIVERSE")
    print(f"{'═' * 60}")
    downloaded = download_datasets(dry_run=args.dry_run)

    if args.dry_run:
        print("\n[DRY RUN] No datasets were downloaded.")
        return

    if not downloaded:
        print("\nERROR: No datasets were successfully downloaded.")
        print("    Check your API key and internet connection.")
        sys.exit(1)if not args.no_merge:
        stats, total = merge_datasets(downloaded)
        # Fix classes with zero or too-few val images (e.g. wild_boar=0, dog=183)
        rebalance_val_split(val_min_annotations=300, move_fraction=0.15)
        generate_data_yaml()
        print_statistics(stats, total)
    else:
        print("\n[SKIP] Merge step skipped (--no-merge flag)")

    print(f"\n{'═' * 60}")
    print("  DATA COLLECTION COMPLETE")
    print(f"  Next step: python augment_nightvision.py")
    print(f"{'═' * 60}\n")

if __name__ == "__main__":
    main()
