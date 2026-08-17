"""
augment_weak_classes.py - Offline augmentation for data-starved training classes.

PHASE 6: Targets only Buffalo (8x) and Cow (4x).
  Buffalo is the chronic weakest class (59.1% mAP in the 350-epoch baseline).
  Cow is the second weakest (76.0%) and is frequently confused with Buffalo.
  Elephant is 92.4% in the baseline — NOT included in Phase 6 augmentation.

Creates GENUINELY DIFFERENT images using 8 transform types:
  flip           - horizontal mirror
  bright         - random brightness + contrast
  hsv            - random hue / saturation / value shift
  crop           - random sub-region crop then resize back (72-92%)
  zoom           - zoom into centre (60-80% of frame)
  flip_crop      - flip then crop
  crop_bright    - crop then brightness change
  zoom_hsv       - zoom then HSV shift

Bounding boxes are recalculated for every geometric transform.
Boxes that lose >70% of their area after cropping are discarded.

Val images are NEVER touched - only training images are augmented.

Usage:
  python augment_weak_classes.py
  python augment_weak_classes.py --classes buffalo cow
  python augment_weak_classes.py --classes buffalo cow --multiplier 8 4
  python augment_weak_classes.py --dry-run
"""

import argparse
import random
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    import os
    os.system(f"{sys.executable} -m pip install opencv-python-headless numpy")
    import cv2
    import numpy as np

from config import MERGED_DIR, TARGET_CLASSES

MIN_BOX_VISIBILITY = 0.30

def read_yolo_labels(path):
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes

def write_yolo_labels(path, boxes):
    with open(path, "w") as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

def imread_unicode(path):
    """
    cv2.imread() returns None on paths with non-ASCII characters (e.g.
    Chinese 文件) on Windows. This wrapper uses np.fromfile + imdecode
    which works regardless of path encoding.
    """
    buf = np.fromfile(str(path), dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def imwrite_unicode(path, img):
    """
    cv2.imwrite() silently fails on non-ASCII paths on Windows.
    Uses imencode + buf.tofile() instead.
    Always saves as JPEG at quality 92 to control disk usage.
    PNG augmented images can be 5-10x larger than needed for YOLO training.
    """
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(path))
    return ok

def aug_flip(img, boxes):
    """Horizontal mirror. x_center becomes 1 - x_center."""
    return cv2.flip(img, 1), [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in boxes]

def aug_crop(img, boxes, lo=0.72, hi=0.92):
    """
    Crop a random sub-region (lo..hi fraction of W and H), resize back.
    Simulates different camera framing or slight zoom in.
    Boxes clipped by >70% are dropped.
    """
    H, W = img.shape[:2]
    r = random.uniform(lo, hi)
    nW, nH = int(W * r), int(H * r)
    x0 = random.randint(0, W - nW)
    y0 = random.randint(0, H - nH)
    x1 = x0 + nW
    y1 = y0 + nH
    out_img = cv2.resize(img[y0:y1, x0:x1], (W, H), interpolation=cv2.INTER_LINEAR)
    new_boxes = []
    for cls, xc, yc, bw, bh in boxes:
        px_xc = xc * W
        px_yc = yc * H
        px_bw = bw * W
        px_bh = bh * H
        orig_area = px_bw * px_bh
        bx1 = px_xc - px_bw / 2
        by1 = px_yc - px_bh / 2
        bx2 = px_xc + px_bw / 2
        by2 = px_yc + px_bh / 2
        cx1 = max(bx1, x0) - x0
        cy1 = max(by1, y0) - y0
        cx2 = min(bx2, x1) - x0
        cy2 = min(by2, y1) - y0
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if orig_area > 0 and (cx2 - cx1) * (cy2 - cy1) / orig_area < MIN_BOX_VISIBILITY:
            continue
        nxc = max(0.0, min(1.0, (cx1 + cx2) / 2 / nW))
        nyc = max(0.0, min(1.0, (cy1 + cy2) / 2 / nH))
        nbw = max(0.001, min(1.0, (cx2 - cx1) / nW))
        nbh = max(0.001, min(1.0, (cy2 - cy1) / nH))
        new_boxes.append([cls, nxc, nyc, nbw, nbh])
    return out_img, new_boxes

def aug_zoom(img, boxes):
    """Zoom into centre region (60-80% of frame). Simulates close-up shots."""
    return aug_crop(img, boxes, lo=0.60, hi=0.80)

def aug_brightness(img, boxes):
    """Random brightness and contrast. No bbox update needed."""
    alpha = random.uniform(0.55, 1.45)
    beta = random.randint(-50, 50)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta), boxes

def aug_hsv(img, boxes):
    """Random hue, saturation, value shift. Simulates different lighting."""
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    h[:, :, 0] = (h[:, :, 0] * random.uniform(0.85, 1.15)).clip(0, 179)
    h[:, :, 1] = (h[:, :, 1] * random.uniform(0.60, 1.40)).clip(0, 255)
    h[:, :, 2] = (h[:, :, 2] * random.uniform(0.60, 1.40)).clip(0, 255)
    return cv2.cvtColor(h.astype("uint8"), cv2.COLOR_HSV2BGR), boxes

def aug_flip_crop(img, boxes):
    img2, b2 = aug_flip(img, boxes)
    return aug_crop(img2, b2)

def aug_crop_bright(img, boxes):
    img2, b2 = aug_crop(img, boxes)
    return aug_brightness(img2, b2)

def aug_zoom_hsv(img, boxes):
    img2, b2 = aug_zoom(img, boxes)
    return aug_hsv(img2, b2)

AUGMENTATIONS = [
    ("flip",       aug_flip),
    ("bright",     aug_brightness),
    ("hsv",        aug_hsv),
    ("crop",       aug_crop),
    ("zoom",       aug_zoom),
    ("flip_crop",  aug_flip_crop),
    ("crop_bright",aug_crop_bright),
    ("zoom_hsv",   aug_zoom_hsv),
]

def _next_counter(images_dir):
    existing = list(images_dir.glob("agd_*.jpg")) + list(images_dir.glob("agd_*.png"))
    if not existing:
        return 1
    return max(int(p.stem.split("_")[1]) for p in existing) + 1

def build_class_index(train_lbls, train_imgs):
    """
    Single-pass index: read every label file ONCE and build
    {class_idx -> [(img_path, lbl_path), ...]} mapping.
    Much faster than scanning once per class for large datasets.
    """
    index = {}
    ext_list = (".jpg", ".jpeg", ".png", ".bmp")
    for lbl in train_lbls.glob("*.txt"):
        with open(lbl) as f:
            ids_in_file = set()
            for ln in f:
                parts = ln.strip().split()
                if len(parts) == 5:
                    ids_in_file.add(int(parts[0]))
        if not ids_in_file:
            continue
        # Find matching image
        img_p = None
        for ext in ext_list:
            candidate = train_imgs / (lbl.stem + ext)
            if candidate.exists():
                img_p = candidate
                break
        if img_p is None:
            continue
        for cid in ids_in_file:
            index.setdefault(cid, []).append((img_p, lbl))
    return index

def augment_class(cls_name, multiplier, dry_run=False, _index=None):

    if cls_name not in TARGET_CLASSES:
        print(f"  {cls_name!r} not in TARGET_CLASSES - skipping")
        return 0

    cls_idx = TARGET_CLASSES.index(cls_name)
    train_imgs = MERGED_DIR / "train" / "images"
    train_lbls = MERGED_DIR / "train" / "labels"

    # Use pre-built index if available (much faster for multiple classes)
    if _index is not None:
        sources = _index.get(cls_idx, [])
    else:
        # Fallback: single-class scan (slower)
        sources = []
        for lbl in train_lbls.glob("*.txt"):
            with open(lbl) as f:
                ids = [int(ln.split()[0]) for ln in f if ln.strip() and len(ln.split()) == 5]
            if cls_idx not in ids:
                continue
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                ip = train_imgs / (lbl.stem + ext)
                if ip.exists():
                    sources.append((ip, lbl))
                    break

    print(f"\n  [{cls_name}]")
    print(f"    Source images containing class : {len(sources)}")
    print(f"    New images to create           : {len(sources) * multiplier}  ({multiplier}x per source)")
    if dry_run:
        return 0

    counter = _next_counter(train_imgs)
    created = 0
    skip_bad_img = 0
    skip_no_boxes = 0

    for img_path, lbl_path in sources:
        img = imread_unicode(img_path)   # Unicode-safe read
        if img is None:
            skip_bad_img += 1
            continue
        boxes = read_yolo_labels(str(lbl_path))
        if not boxes:
            skip_bad_img += 1
            continue

        pool = (AUGMENTATIONS * ((multiplier // len(AUGMENTATIONS)) + 1))[:]
        chosen = random.sample(pool, multiplier)

        for aug_name, aug_fn in chosen:
            try:
                aug_img, aug_boxes = aug_fn(img.copy(), [b[:] for b in boxes])
            except Exception as e:
                print(f"    {aug_name} failed: {e}")
                skip_bad_img += 1
                continue
            if not aug_boxes:
                skip_no_boxes += 1
                continue

            stem = f"agd_{counter:06d}"
            # Always save augmented images as .jpg (JPEG q92) regardless of source extension.
            # This prevents PNG-sourced augmented images from being 5-10x larger than needed.
            imwrite_unicode(train_imgs / (stem + ".jpg"), aug_img)
            write_yolo_labels(str(train_lbls / (stem + ".txt")), aug_boxes)
            counter += 1
            created += 1

    print(f"    Created: {created}")
    if skip_bad_img:  print(f"    Skipped (unreadable image)  : {skip_bad_img}")
    if skip_no_boxes: print(f"    Skipped (all boxes cropped) : {skip_no_boxes}")
    return created

def main():
    random.seed(42)
    np.random.seed(42)
    parser = argparse.ArgumentParser(
        description=(
            "Augment weak training classes with genuinely different images. "
            "Val set is NEVER modified."
        )
    )
    parser.add_argument("--classes", nargs="+", default=["buffalo", "cow"],
                        help="Classes to augment (default: buffalo cow — the two weak classes at 350-epoch baseline)")
    parser.add_argument("--multiplier", nargs="+", type=int, default=[8, 4],
                        help="New images per source image. Either one value for all classes, "
                             "or one value per class. "
                             "(default: 8 4 = buffalo 8x, cow 4x — buffalo is the worst class)") 
    parser.add_argument("--dry-run", action="store_true",
                        help="Count only, do not write files")
    args = parser.parse_args()

    # Resolve multipliers: broadcast single value or match per-class
    if len(args.multiplier) == 1:
        multipliers = args.multiplier * len(args.classes)
    elif len(args.multiplier) == len(args.classes):
        multipliers = args.multiplier
    else:
        parser.error(f"--multiplier must have 1 value or {len(args.classes)} values (one per class)")

    print("\n" + "=" * 55)
    print("  OFFLINE AUGMENTATION FOR WEAK CLASSES")
    print("=" * 55)
    print(f"  Classes    : {args.classes}")
    print(f"  Multipliers: {multipliers}x  (one per class)")
    out_path = str(MERGED_DIR / "train").encode("ascii", errors="replace").decode("ascii")
    print(f"  Output     : {out_path}  (val is untouched)")
    if args.dry_run:
        print("  MODE       : DRY RUN - no files written")
    print("=" * 55)

    train_imgs = MERGED_DIR / "train" / "images"
    train_lbls = MERGED_DIR / "train" / "labels"

    print("  Building class index (single pass through all label files)...")
    index = build_class_index(train_lbls, train_imgs)
    for cls_id, name in enumerate(TARGET_CLASSES):
        cnt = len(index.get(cls_id, []))
        if cnt > 0:
            print(f"    class {cls_id:2d} {name:<18} : {cnt} images")
    print()

    total = 0
    for cls, mult in zip(args.classes, multipliers):
        total += augment_class(cls, mult, args.dry_run, _index=index)

    print(f"\n  Total new training images : {total}")
    if args.dry_run:
        print("  (DRY RUN - no files written)")
    print("\n  Next step: repack the dataset zip and start training on Kaggle.")
    print("=" * 55 + "\n")

if __name__ == "__main__":
    main()