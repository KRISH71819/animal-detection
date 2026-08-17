"""
hard_negative_mining.py -- Cow / Buffalo confusion analyser (Phase 6)
=======================================================================
Runs the YOLO11m 350-epoch model over the validation set and finds every
image where the model gets Cow or Buffalo wrong.

Three error buckets are saved to data/hard_negatives/:
  buffalo_as_cow/  -- GT=buffalo, predicted=cow
  cow_as_buffalo/  -- GT=cow,     predicted=buffalo
  buffalo_missed/  -- GT=buffalo, model predicted nothing (recall gap)

Usage:
  python hard_negative_mining.py
  python hard_negative_mining.py --weights runs-10class-350/animal_guard_train/weights/best.pt
  python hard_negative_mining.py --conf 0.25 --iou 0.4
  python hard_negative_mining.py --copy-images
"""

import argparse
import shutil
import sys
from pathlib import Path
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    import os; os.system(f"{sys.executable} -m pip install ultralytics")
    from ultralytics import YOLO

from config import TARGET_CLASSES, MERGED_DIR, RUNS_DIR

COW_IDX     = TARGET_CLASSES.index("cow")
BUFFALO_IDX = TARGET_CLASSES.index("buffalo")

HARD_NEG_DIR = Path("data") / "hard_negatives"
BUCKETS = {
    "buffalo_as_cow": HARD_NEG_DIR / "buffalo_as_cow",
    "cow_as_buffalo": HARD_NEG_DIR / "cow_as_buffalo",
    "buffalo_missed":  HARD_NEG_DIR / "buffalo_missed",
}

def find_best_weights():
    candidates = [
        Path("runs-10class-350") / "animal_guard_train" / "weights" / "best.pt",
        Path("runs-10class-350") / "animal_guard_train" / "weights" / "last.pt",
        RUNS_DIR / "animal_guard_train" / "weights" / "best.pt",
        Path("best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def load_gt_boxes(label_path):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append((int(parts[0]), *[float(x) for x in parts[1:]]))
    return boxes

def box_iou(b1, b2):
    def to_xyxy(b):
        xc, yc, bw, bh = b
        return xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2
    x1, y1, x2, y2 = to_xyxy(b1)
    x3, y3, x4, y4 = to_xyxy(b2)
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = (x2 - x1)*(y2 - y1) + (x4 - x3)*(y4 - y3) - inter
    return inter / u if u > 0 else 0.0

def analyse_val_set(weights_path, conf, iou_thresh, copy_images):
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  HARD NEGATIVE MINING  --  Cow / Buffalo Confusion")
    print(SEP)
    print(f"  Weights : {weights_path}")
    print(f"  Conf    : {conf}   IOU : {iou_thresh}\n")

    model = YOLO(str(weights_path))
    val_imgs = MERGED_DIR / "val" / "images"
    val_lbls = MERGED_DIR / "val" / "labels"
    img_paths = sorted(
        list(val_imgs.glob("*.jpg")) +
        list(val_imgs.glob("*.jpeg")) +
        list(val_imgs.glob("*.png"))
    )
    if not img_paths:
        print("No validation images found. Run collect_data.py first.")
        sys.exit(1)
    print(f"  Analysing {len(img_paths)} validation images...")
    for d in BUCKETS.values():
        d.mkdir(parents=True, exist_ok=True)

    results_by_bucket = defaultdict(list)
    stats = defaultdict(int)
    IOU_MATCH = 0.3

    for img_path in img_paths:
        lbl_path = val_lbls / (img_path.stem + ".txt")
        gt_boxes = load_gt_boxes(lbl_path)
        gt_cow_b     = [(b[1], b[2], b[3], b[4]) for b in gt_boxes if b[0] == COW_IDX]
        gt_buffalo_b = [(b[1], b[2], b[3], b[4]) for b in gt_boxes if b[0] == BUFFALO_IDX]
        if not gt_cow_b and not gt_buffalo_b:
            continue
        stats["img_with_cow_or_buffalo"] += 1
        stats["gt_cow"]     += len(gt_cow_b)
        stats["gt_buffalo"] += len(gt_buffalo_b)

        res = model.predict(str(img_path), conf=conf, iou=iou_thresh, verbose=False, device="cpu")[0]
        pred_boxes = res.boxes
        pred_list = []
        if pred_boxes is not None and len(pred_boxes) > 0:
            pred_list = [
                (int(pred_boxes.cls[i].item()),
                 pred_boxes.xywhn[i].tolist(),
                 float(pred_boxes.conf[i].item()))
                for i in range(len(pred_boxes))
            ]
        pred_cow_b     = [(b[1], b[2]) for b in pred_list if b[0] == COW_IDX]
        pred_buffalo_b = [(b[1], b[2]) for b in pred_list if b[0] == BUFFALO_IDX]

        # Bucket 1: buffalo predicted as cow
        for gt_box in gt_buffalo_b:
            matched_as_cow = any(box_iou(gt_box, pb[0]) >= IOU_MATCH for pb in pred_cow_b)
            not_as_buffalo = not any(box_iou(gt_box, pb[0]) >= IOU_MATCH for pb in pred_buffalo_b)
            if matched_as_cow and not_as_buffalo:
                conf_val = max((pb[1] for pb in pred_cow_b), default=0.0)
                results_by_bucket["buffalo_as_cow"].append((img_path, lbl_path, conf_val))
                stats["buffalo_as_cow"] += 1
                break

        # Bucket 2: cow predicted as buffalo
        for gt_box in gt_cow_b:
            matched_as_buffalo = any(box_iou(gt_box, pb[0]) >= IOU_MATCH for pb in pred_buffalo_b)
            not_as_cow         = not any(box_iou(gt_box, pb[0]) >= IOU_MATCH for pb in pred_cow_b)
            if matched_as_buffalo and not_as_cow:
                conf_val = max((pb[1] for pb in pred_buffalo_b), default=0.0)
                results_by_bucket["cow_as_buffalo"].append((img_path, lbl_path, conf_val))
                stats["cow_as_buffalo"] += 1
                break

        # Bucket 3: buffalo completely missed
        if gt_buffalo_b and not pred_buffalo_b and not any(
            box_iou(gt_box, pb[0]) >= IOU_MATCH
            for gt_box in gt_buffalo_b for pb in pred_cow_b
        ):
            results_by_bucket["buffalo_missed"].append((img_path, lbl_path, 0.0))
            stats["buffalo_missed"] += 1

    return results_by_bucket, stats

def save_results(results_by_bucket, stats, copy_images):
    SEP = "-" * 60
    print(f"\n{SEP}")
    print("  RESULTS SUMMARY")
    print(SEP)
    print(f"  Val images with Cow or Buffalo : {stats['img_with_cow_or_buffalo']}")
    print(f"  Ground-truth Cow boxes         : {stats['gt_cow']}")
    print(f"  Ground-truth Buffalo boxes     : {stats['gt_buffalo']}")
    print()

    total = 0
    labels = {
        "buffalo_as_cow": "Buffalo GT -> Cow predicted   (FN buffalo, FP cow)",
        "cow_as_buffalo": "Cow GT -> Buffalo predicted   (FN cow, FP buffalo)",
        "buffalo_missed":  "Buffalo GT -> Missed entirely (FN, low recall)",
    }
    for bucket_name, entries in results_by_bucket.items():
        dest_dir = BUCKETS[bucket_name]
        lbl_dir  = dest_dir / "labels"
        img_dir  = dest_dir / "images"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        if copy_images:
            img_dir.mkdir(parents=True, exist_ok=True)
        avg_conf = (sum(e[2] for e in entries) / len(entries)) if entries else 0.0
        for img_path, lbl_path, _ in entries:
            if lbl_path.exists():
                shutil.copy2(lbl_path, lbl_dir / lbl_path.name)
            if copy_images and img_path.exists():
                shutil.copy2(img_path, img_dir / img_path.name)
        print(f"  [{bucket_name}]  {labels[bucket_name]}")
        print(f"    Count : {len(entries)}  |  Avg confidence : {avg_conf:.3f}")
        print(f"    Saved : {dest_dir}")
        print()
        total += len(entries)

    print(f"  TOTAL ERRORS : {total}")
    report_path = HARD_NEG_DIR / "confusion_report.tsv"
    with open(report_path, "w") as f:
        f.write("bucket\tcount\tavg_conf\n")
        for bucket_name, entries in results_by_bucket.items():
            avg_conf = (sum(e[2] for e in entries) / len(entries)) if entries else 0.0
            f.write(f"{bucket_name}\t{len(entries)}\t{avg_conf:.4f}\n")
    print(f"  Report: {report_path}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Hard negative mining: find Cow/Buffalo confusion in validation set."
    )
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to YOLO weights. Auto-detects 350-epoch YOLO11m best.pt.")
    parser.add_argument("--conf",  type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--iou",   type=float, default=0.40, help="NMS IoU threshold (default 0.40)")
    parser.add_argument("--copy-images", action="store_true",
                        help="Also copy images into hard_negatives/ (default: labels only)")
    args = parser.parse_args()

    if args.weights:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            print(f"[X] Weights not found: {weights_path}")
            sys.exit(1)
    else:
        weights_path = find_best_weights()
        if weights_path is None:
            print("[X] No weights found. Use --weights path/to/best.pt")
            sys.exit(1)
        print(f"Auto-detected weights: {weights_path}")

    results, stats = analyse_val_set(weights_path, args.conf, args.iou, args.copy_images)
    save_results(results, stats, args.copy_images)

if __name__ == "__main__":
    main()
