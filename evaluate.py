import argparse
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np────────────────────────────
from config import (
    DATA_YAML,
    MERGED_DIR,
    RUNS_DIR,
    TARGET_CLASSES,
    NUM_CLASSES,
    INFERENCE_CONFIG,
)

def run_validation(weights_path, imgsz=640, batch=16, device=0):
    """
    Run Ultralytics YOLO validation on the full validation set.

    Args:
        weights_path: Path to trained model weights
        imgsz: Image size for validation
        batch: Batch size
        device: Device index or "cpu"

    Returns:
        Ultralytics validation results object
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    if not Path(weights_path).exists():
        print(f"ERROR: Weights file not found: {weights_path}")
        sys.exit(1)

    if not DATA_YAML.exists():
        print(f"ERROR: data.yaml not found: {DATA_YAML}")
        print(f"    Run: python collect_data.py")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    print(f"Running validation on: {DATA_YAML}")
    print(f"    Image size: {imgsz}, Batch: {batch}, Device: {device}")

    results = model.val(
        data=str(DATA_YAML),
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=True,
        plots=True,  # Generate confusion matrix, PR curves, etc.
    )

    return model, results

def print_detailed_results(results):
    """Print detailed per-class and overall metrics."""
    print(f"\n{'═' * 70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'═' * 70}")

    # Overall metrics
    box = results.box
    print(f"  │  mAP@0.5:       {box.map50:.4f}                              │")
    print(f"  │  mAP@0.5:0.95:  {box.map:.4f}                              │")
    print(f"  │  Mean Precision: {box.mp:.4f}                              │")
    print(f"  │  Mean Recall:    {box.mr:.4f}                              │")

    # Per-class metrics
    print(f"\n  {'Class':<15} {'AP@0.5':>8} {'AP@.5:.95':>10} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print(f"  {'─' * 15} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}")

    # Access per-class APs
    ap50 = box.ap50           # AP at IoU=0.5 per class
    ap = box.ap               # AP at IoU=0.5:0.95 per class
    p = box.p                 # Precision per class
    r = box.r                 # Recall per class

    for i, cls_name in enumerate(TARGET_CLASSES):
        if i >= len(ap50):
            break

        precision = pif i < len(p) else 0
        recall = rif i < len(r) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Flag poor performers
        flag = ""
        if ap50< 0.3:
            flag = " ⚠️ LOW"
        elif ap50< 0.5:
            flag = " ⚡"

        print(f"  {cls_name:<15} {ap50[i]:>8.4f} {ap[i]:>10.4f} "
              f"{precision:>8.4f} {recall:>8.4f} {f1:>8.4f}{flag}")

    print(f"  {'─' * 15} {'─' * 8} {'─' * 10} {'─' * 8} {'─' * 8} {'─' * 8}")

def analyze_day_vs_night(weights_path):
    """
    Analyze model performance separately on day images vs synthetic IR images.

    Day images: filenames without '_ir_mild' or '_ir_harsh'
    Night images: filenames with '_ir_mild' or '_ir_harsh'
    """
    import cv2

    try:
        from ultralytics import YOLO
    except ImportError:
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    val_images_dir = MERGED_DIR / "val" / "images"
    val_labels_dir = MERGED_DIR / "val" / "labels"

    if not val_images_dir.exists():
        print("Validation images not found. Skipping day/night analysis.")
        return

    # Separate images by type
    day_images = []
    night_images = []

    for img_path in val_images_dir.iterdir():
        if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            if "_ir_mild" in img_path.stem or "_ir_harsh" in img_path.stem:
                night_images.append(img_path)
            else:
                day_images.append(img_path)

    if not night_images:
        print("No synthetic IR images in validation set.")
        print("    Run augment_nightvision.py with --ratio applied to val set,")
        print("    or skip this analysis.")
        return

    model = YOLO(weights_path)
    conf = INFERENCE_CONFIG["conf_threshold"]

    def _eval_subset(image_paths, subset_name):
        """Evaluate model on a subset of images."""
        total = 0
        correct = 0
        class_counts = defaultdict(int)
        class_detected = defaultdict(int)

        for img_path in image_paths:
            label_path = val_labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Count ground truth
            with open(label_path) as f:
                gt_classes = set()
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        if cls_idx < NUM_CLASSES:
                            gt_classes.add(cls_idx)
                            class_counts[TARGET_CLASSES[cls_idx]] += 1

            total += len(gt_classes)

            # Run inference
            results = model.predict(
                source=str(img_path), conf=conf, verbose=False
            )
            pred_classes = set()
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        pred_classes.add(int(box.cls.item()))

            # Count correct class predictions
            hits = gt_classes & pred_classes
            correct += len(hits)
            for cls_idx in hits:
                if cls_idx < NUM_CLASSES:
                    class_detected[TARGET_CLASSES[cls_idx]] += 1

        recall = correct / total if total > 0 else 0
        print(f"  │  Class-level recall: {recall:.4f} ({correct}/{total})  │")

        for cls in TARGET_CLASSES:
            gt = class_counts.get(cls, 0)
            det = class_detected.get(cls, 0)
            if gt > 0:
                cls_recall = det / gt
                print(f"  │    {cls:<12}: {cls_recall:.2f} ({det}/{gt})")

    _eval_subset(day_images, "DAY (Original)")
    _eval_subset(night_images, "NIGHT (Synthetic IR)")

def run_speed_benchmark(weights_path, num_images=50, imgsz=640):
    """
    Benchmark inference speed on sample images.

    Args:
        weights_path: Path to trained weights
        num_images: Number of images to benchmark
        imgsz: Inference image size
    """
    import cv2

    try:
        from ultralytics import YOLO
    except ImportError:
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    val_images_dir = MERGED_DIR / "val" / "images"
    if not val_images_dir.exists():
        print("Validation images not found. Skipping benchmark.")
        return

    images = list(val_images_dir.glob("*"))
    images = [f for f in images if f.suffix.lower() in {".jpg", ".jpeg", ".png"}][:num_images]

    if not images:
        print("No images found for benchmarking.")
        return

    model = YOLO(weights_path)
    conf = INFERENCE_CONFIG["conf_threshold"]

    # Warmup
    print(f"\n  Speed benchmark ({len(images)} images, {imgsz}px)")
    print(f"  Warming up (3 iterations)...")
    for _ in range(3):
        img = cv2.imread(str(images[0]))
        model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)

    # Benchmark
    times = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        start = time.perf_counter()
        model.predict(source=img, conf=conf, imgsz=imgsz, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    if times:
        times = np.array(times)
        print(f"  │  Mean:    {np.mean(times):>8.1f} ms                         │")
        print(f"  │  Median:  {np.median(times):>8.1f} ms                         │")
        print(f"  │  Std:     {np.std(times):>8.1f} ms                         │")
        print(f"  │  Min:     {np.min(times):>8.1f} ms                         │")
        print(f"  │  Max:     {np.max(times):>8.1f} ms                         │")
        print(f"  │  FPS:     {1000 / np.mean(times):>8.1f}                            │")

def save_results_json(results, weights_path, save_dir):
    """Save evaluation results as JSON for tracking experiments."""
    output = {
        "weights": str(weights_path),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "mean_precision": float(results.box.mp),
        "mean_recall": float(results.box.mr),
        "per_class": {},
    }

    for i, cls_name in enumerate(TARGET_CLASSES):
        if i >= len(results.box.ap50):
            break
        output["per_class"][cls_name] = {
            "ap50": float(results.box.ap50[i]),
            "ap50_95": float(results.box.ap[i]),
            "precision": float(results.box.p[i]) if i < len(results.box.p) else 0,
            "recall": float(results.box.r[i]) if i < len(results.box.r) else 0,
        }

    json_path = save_dir / "eval_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

def mine_failures(weights_path, conf=0.25, max_per_bucket=30):
    """
    Scan the validation set and export images where the model makes specific
    confusion errors (buffalo↔cow, dog↔wild_boar) or misses a class entirely.

    Output folders are written to runs/eval/failures/<category>/.
    Open the folders and eyeball whether it's a label-noise issue or a
    genuine model failure.

    Args:
        weights_path: Path to trained weights
        conf: Confidence threshold for predictions
        max_per_bucket: Max images exported per error category
    """
    import shutil
    import cv2

    try:
        from ultralytics import YOLO
    except ImportError:
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    val_images_dir = MERGED_DIR / "val" / "images"
    val_labels_dir = MERGED_DIR / "val" / "labels"
    failure_dir    = RUNS_DIR / "eval" / "failures"

    if not val_images_dir.exists():
        print("Validation images not found. Skipping failure mining.")
        return

    # ── Confusion pairs to track: (gt_class_idx, predicted_class_idx) ──
    CONFUSION_PAIRS = {
        (1, 0): "buffalo_predicted_as_cow",
        (0, 1): "cow_predicted_as_buffalo",
        (3, 4): "dog_predicted_as_wild_boar",
        (4, 3): "wild_boar_predicted_as_dog",
    }
    # Missed-detection buckets for the two weakest classes
    MISSED_BUCKETS = {
        0: "cow_missed",
        1: "buffalo_missed",
        3: "dog_missed",
    }

    model  = YOLO(weights_path)
    counts = defaultdict(int)
    images = [p for p in val_images_dir.iterdir()
              if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]

    print(f"\n  Mining failures across {len(images)} val images (conf={conf}) …")
    print(f"  Max {max_per_bucket} images per error category")
    print(f"  Output → {failure_dir}\n")

    for img_path in images:
        label_path = val_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        # Ground-truth classes in this image
        gt_classes = set()
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    gt_classes.add(int(parts[0]))

        # Model predictions
        result = model.predict(source=str(img_path), conf=conf, verbose=False)
        pred_classes = set()
        for r in result:
            if r.boxes is not None:
                for box in r.boxes:
                    pred_classes.add(int(box.cls.item()))

        # ── Check confusion pairs ──
        for (gt_cls, pred_cls), bucket in CONFUSION_PAIRS.items():
            if counts[bucket] >= max_per_bucket:
                continue
            # GT has gt_cls, model predicts pred_cls but NOT gt_cls
            if gt_cls in gt_classes and pred_cls in pred_classes and gt_cls not in pred_classes:
                out_dir = failure_dir / bucket
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, out_dir / img_path.name)
                counts[bucket] += 1for gt_cls, bucket in MISSED_BUCKETS.items():
            if counts[bucket] >= max_per_bucket:
                continue
            if gt_cls in gt_classes and gt_cls not in pred_classes:
                out_dir = failure_dir / bucket
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(img_path, out_dir / img_path.name)
                counts[bucket] += 1print(f"  {'─' * 45}")
    print(f"  {'Category':<35} {'Images':>6}")
    print(f"  {'─' * 45}")
    for bucket, n in sorted(counts.items()):
        flag = "  ⚠️  HIGH" if n >= max_per_bucket else ""
        print(f"  {bucket:<35} {n:>6}{flag}")
    print(f"  {'─' * 45}")
    print(f"\n  Open {failure_dir} to inspect the exported images.")
    print(f"  If you see clearly-correct labels predicted as the wrong class,")
    print(f"  the training data has label noise — relabeling is the fix.")
    print(f"  If images are genuinely ambiguous, more diverse data is needed.")

def conf_sweep(weights_path, confs=None, imgsz=640, batch=4, device="cpu"):
    """
    Run model.val() at multiple confidence thresholds to find the per-class
    optimal conf that maximises F1-score.

    NOTE: Each val run on CPU takes ~20-30 min for 3,600 images.
          Limit confs list or use --device 0 for speed.

    Args:
        weights_path: Path to trained weights
        confs: List of confidence thresholds to sweep (default: [0.1, 0.25, 0.5])
        imgsz: Image size
        batch: Batch size
        device: Device ('cpu' or '0')
    """
    if confs is None:
        confs = [0.1, 0.25, 0.5]

    try:
        from ultralytics import YOLO
    except ImportError:
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    model = YOLO(weights_path)

    print(f"\n  {'─' * 65}")
    print(f"  CONFIDENCE SWEEP  (thresholds: {confs})")
    print(f"  {'─' * 65}")

    best_conf = {}   # class → (best_f1, best_conf)
    all_rows  = []   # (conf, class, P, R, F1)

    for conf in confs:
        print(f"\n  Running val at conf={conf} …")
        results = model.val(
            data=str(DATA_YAML),
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            verbose=False,
            plots=False,
        )
        box = results.box
        print(f"  conf={conf}  mAP50={box.map50:.4f}  P={box.mp:.4f}  R={box.mr:.4f}")
        print(f"  {'Class':<14} {'P':>6} {'R':>6} {'F1':>6}")
        print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*6}")
        for i, cls_name in enumerate(TARGET_CLASSES):
            if i >= len(box.ap50):
                break
            p  = float(box.p[i]) if i < len(box.p) else 0.0
            r  = float(box.r[i]) if i < len(box.r) else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            print(f"  {cls_name:<14} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")
            all_rows.append((conf, cls_name, p, r, f1))
            if f1 > best_conf.get(cls_name, (-1, None))[0]:
                best_conf[cls_name] = (f1, conf)

    # ── Best conf per class ──
    print(f"\n  {'─' * 50}")
    print(f"  RECOMMENDED PER-CLASS CONFIDENCE THRESHOLDS")
    print(f"  (conf that maximised F1 in this sweep)")
    print(f"  {'─' * 50}")
    print(f"  {'Class':<14} {'Best conf':>9} {'Best F1':>8}")
    print(f"  {'─'*14} {'─'*9} {'─'*8}")
    for cls_name in TARGET_CLASSES:
        if cls_name in best_conf:
            f1, c = best_conf[cls_name]
            print(f"  {cls_name:<14} {c:>9.2f} {f1:>8.3f}")
    print(f"  {'─' * 50}")
    print(f"\n  Tip: use these thresholds in detect.py with --conf <value>")
    print(f"  or implement per-class conf filtering in the pipeline.")

def main():
    parser = argparse.ArgumentParser(
        description="Animal Guard — Model evaluation and benchmarking"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained YOLO weights (.pt)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size for validation (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size for validation (default: 16)"
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="Device: 0 for GPU, cpu for CPU"
    )
    parser.add_argument(
        "--save-plots", action="store_true",
        help="Save confusion matrix and PR curve plots"
    )
    parser.add_argument(
        "--benchmark", type=int, nargs="?", const=50, default=None,
        help="Run speed benchmark on N images (default: 50)"
    )
    parser.add_argument(
        "--day-night", action="store_true",
        help="Analyze performance breakdown: day vs synthetic IR images"
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save results as JSON for experiment tracking"
    )
    parser.add_argument(
        "--mine-failures", action="store_true",
        help="Export val images where buffalo↔cow / dog↔wild_boar confusion occurs, and missed detections"
    )
    parser.add_argument(
        "--mine-conf", type=float, default=0.25,
        help="Confidence threshold for --mine-failures (default: 0.25)"
    )
    parser.add_argument(
        "--mine-max", type=int, default=30,
        help="Max images to export per failure category (default: 30)"
    )
    parser.add_argument(
        "--conf-sweep", action="store_true",
        help="Sweep multiple confidence thresholds to find optimal per-class conf (slow on CPU)"
    )
    parser.add_argument(
        "--sweep-confs", type=float, nargs="+", default=[0.1, 0.25, 0.5],
        help="Confidence values to sweep (default: 0.1 0.25 0.5)"
    )
    args = parser.parse_args()

    # ── Run validation ──
    print(f"\n{'═' * 60}")
    print(f"  ANIMAL GUARD MODEL EVALUATION")
    print(f"{'═' * 60}")

    model, results = run_validation(
        args.weights,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # ── Print results ──
    print_detailed_results(results)if args.save_json:
        save_dir = RUNS_DIR / "eval"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_results_json(results, args.weights, save_dir)

    # ── Day vs Night analysis ──
    if args.day_night:
        analyze_day_vs_night(args.weights)

    # ── Speed benchmark ──
    if args.benchmark is not None:
        run_speed_benchmark(args.weights, num_images=args.benchmark, imgsz=args.imgsz)

    # ── Mine failure cases ──
    if args.mine_failures:
        mine_failures(
            args.weights,
            conf=args.mine_conf,
            max_per_bucket=args.mine_max,
        )

    # ── Confidence sweep ──
    if args.conf_sweep:
        conf_sweep(
            args.weights,
            confs=args.sweep_confs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    print(f"\n{'═' * 60}")
    print(f"  EVALUATION COMPLETE")
    print(f"{'═' * 60}\n")

if __name__ == "__main__":
    main()
