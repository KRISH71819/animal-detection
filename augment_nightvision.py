import argparse
import random
import sys
from pathlib import Path

# Reconfigure stdout/stderr to use UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    print("albumentations not found. Installing...")
    import os
    os.system(f"{sys.executable} -m pip install albumentations")
    import albumentations as A
from config import (
    MERGED_DIR,
    IR_AUGMENT_RATIO,
    IR_SETTINGS,
)

def build_ir_pipeline(severity="mild"):
    """
    Build an Albumentations pipeline that simulates a cheap IR camera feed.

    Args:
        severity: "mild" (clear night, good IR) or "harsh" (foggy, noisy, bad IR)

    Returns:
        A.Compose pipeline
    """
    s = IR_SETTINGS[severity]

    transforms = [
        # Stage 1: Convert to grayscale (IR sensors are single-channel)
        A.ToGray(p=1.0),

        # Stage 2: Gamma correction — darken mid-tones to simulate night
        A.RandomGamma(
            gamma_limit=(
                int(s["gamma_range"][0] * 100),
                int(s["gamma_range"][1] * 100)
            ),
            p=0.9,
        ),

        # Stage 3: CLAHE — simulate IR sensor auto-gain/auto-exposure
        A.CLAHE(
            clip_limit=s["clahe_clip_limit"],
            tile_grid_size=(8, 8),
            p=0.8,
        ),

        # Stage 4a: Random brightness & contrast shifts
        A.RandomBrightnessContrast(
            brightness_limit=s["brightness_limit"],
            contrast_limit=s["contrast_limit"],
            p=0.7,
        ),

        # Stage 4b: Gaussian sensor noise
        A.GaussNoise(
            std_range=s["gauss_noise_std"],
            p=0.85,
        ),

        # Stage 4c: ISO noise (color-correlated noise patterns)
        A.ISONoise(
            intensity=s["iso_noise_intensity"],
            p=0.6,
        ),

        # Stage 5a: Slight blur (simulates focus softness)
        A.Blur(
            blur_limit=s["blur_limit"],
            p=0.3,
        ),

        # Stage 5b: Vignette (darker corners from cheap lenses)
        # Simulated by applying a radial gradient mask
    ]

    return A.Compose(transforms)

def apply_vignette(image, strength=0.5):
    """
    Apply a vignette effect (darker corners) to simulate cheap lens optics.

    Args:
        image: Input image (numpy array)
        strength: 0.0 (no effect) to 1.0 (strong darkening at edges)

    Returns:
        Image with vignette applied
    """
    h, w = image.shape[:2]

    # Create radial gradient
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx ** 2 + yy ** 2)

    # Normalize and invert (center=1, edges=0)
    radius = np.clip(radius, 0, 1)
    vignette_mask = 1.0 - (radius * strength)
    vignette_mask = np.clip(vignette_mask, 0.3, 1.0)

    # Apply mask
    if len(image.shape) == 3:
        vignette_mask = np.stack([vignette_mask] * image.shape[2], axis=-1)

    result = (image.astype(np.float32) * vignette_mask).astype(np.uint8)
    return result

def augment_image(image, severity="mild"):
    """
    Apply the full synthetic IR augmentation pipeline to a single image.

    Args:
        image: Input BGR image (numpy array)
        severity: "mild" or "harsh"

    Returns:
        Augmented image (numpy array, same shape)
    """
    pipeline = build_ir_pipeline(severity)
    augmented = pipeline(image=image)["image"]

    # Apply vignette separately (not in albumentations by default)
    s = IR_SETTINGS[severity]
    if random.random() < s["vignette_p"]:
        augmented = apply_vignette(augmented, strength=random.uniform(0.3, 0.7))

    return augmented

def preview_augmentation(num_samples=5):
    """
    Show side-by-side previews of original vs augmented images.
    Opens a window or saves preview images to data/augmented/preview/.
    """
    train_images = MERGED_DIR / "train" / "images"
    if not train_images.exists():
        print("ERROR: No merged training images found. Run collect_data.py first.")
        sys.exit(1)

    image_files = list(train_images.glob("*"))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [f for f in image_files if f.suffix.lower() in image_extensions]

    if not image_files:
        print("ERROR: No images found in training directory.")
        sys.exit(1)

    samples = random.sample(image_files, min(num_samples, len(image_files)))
    preview_dir = MERGED_DIR.parent / "augmented" / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(samples):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        # Generate both mild and harsh versions
        mild = augment_image(img, severity="mild")
        harsh = augment_image(img, severity="harsh")

        # Ensure all images have same number of channels for concatenation
        if len(mild.shape) == 2:
            mild = cv2.cvtColor(mild, cv2.COLOR_GRAY2BGR)
        if len(harsh.shape) == 2:
            harsh = cv2.cvtColor(harsh, cv2.COLOR_GRAY2BGR)

        # Resize all to same height for side-by-side comparison
        h = 400
        aspect = img.shape[1] / img.shape[0]
        w = int(h * aspect)
        img_r = cv2.resize(img, (w, h))
        mild_r = cv2.resize(mild, (w, h))
        harsh_r = cv2.resize(harsh, (w, h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_r, "Original", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(mild_r, "IR Mild", (10, 30), font, 0.8, (0, 255, 0), 2)
        cv2.putText(harsh_r, "IR Harsh", (10, 30), font, 0.8, (0, 255, 0), 2)

        comparison = np.hstack([img_r, mild_r, harsh_r])

        preview_path = preview_dir / f"preview_{i + 1}.jpg"
        is_success, im_buf_arr = cv2.imencode(".jpg", comparison)
        if is_success:
            im_buf_arr.tofile(str(preview_path))
        print(f"  Saved preview: {preview_path}")

    print(f"\n  Previews saved to: {preview_dir}")
    print(f"  Open the preview images to verify IR simulation quality.")

def run_augmentation(ratio=None):
    """
    Augment a percentage of training images with synthetic IR simulation.

    The augmented images are saved ALONGSIDE the originals in the same
    training directory. Label files are duplicated as-is (no spatial
    transforms are applied, so bounding boxes remain valid).
    """
    augment_ratio = ratio if ratio is not None else IR_AUGMENT_RATIO
    random.seed(42)
    np.random.seed(42)

    train_images_dir = MERGED_DIR / "train" / "images"
    train_labels_dir = MERGED_DIR / "train" / "labels"

    if not train_images_dir.exists():
        print("ERROR: No merged training images found. Run collect_data.py first.")
        sys.exit(1)

    # Get all training images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_images = [
        f for f in train_images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]

    # Filter out already-augmented images (avoid double augmentation)
    original_images = [
        f for f in all_images
        if "_ir_mild" not in f.stem and "_ir_harsh" not in f.stem
    ]

    if not original_images:
        print("ERROR: No original training images found.")
        sys.exit(1)

    # Randomly select images for augmentation
    num_to_augment = int(len(original_images) * augment_ratio)
    selected = random.sample(original_images, min(num_to_augment, len(original_images)))

    print(f"\n{'═' * 60}")
    print(f"  SYNTHETIC IR AUGMENTATION")
    print(f"{'═' * 60}")
    print(f"  Total original images: {len(original_images)}")
    print(f"  Augmentation ratio:    {augment_ratio:.0%}")
    print(f"  Images to augment:     {len(selected)}")
    print(f"  New images created:    {len(selected) * 2} (mild + harsh)")
    print(f"{'─' * 60}")

    created_count = 0
    skipped_count = 0

    for img_path in tqdm(selected, desc="  Augmenting", unit="img"):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped_count += 1
            continue

        label_path = train_labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped_count += 1
            continue

        # Read the original label content (will be duplicated for augmented images)
        label_content = label_path.read_text(encoding="utf-8")

        # Generate MILD IR version
        for severity in ["mild", "harsh"]:
            aug_img = augment_image(img, severity=severity)
            suffix = f"_ir_{severity}"

            # Save augmented image as JPEG (quality 92) regardless of source format.
            # PNG sources can be 500KB-2MB each; JPEG at q92 is ~50-150KB.
            # Visually lossless for YOLO training, saves 60-80% disk space on Kaggle.
            aug_img_path = train_images_dir / f"{img_path.stem}{suffix}.jpg"
            is_success, im_buf_arr = cv2.imencode(".jpg", aug_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if is_success:
                im_buf_arr.tofile(str(aug_img_path))

            # Duplicate label file (spatial coords unchanged)
            aug_lbl_path = train_labels_dir / f"{img_path.stem}{suffix}.txt"
            aug_lbl_path.write_text(label_content, encoding="utf-8")

            created_count += 1

    # Final statistics
    total_after = len(list(train_images_dir.glob("*")))
    print(f"\n{'─' * 60}")
    print(f"  Results:")
    print(f"  ├─ Created:    {created_count} augmented images")
    print(f"  ├─ Skipped:    {skipped_count} (unreadable or no labels)")
    print(f"  └─ Total now:  {total_after} images in training set")
    print(f"\n  IR augmentation complete!")
    print(f"  Next step: python train.py")
    print(f"{'═' * 60}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Animal Guard — Synthetic night vision / IR augmentation"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help=f"Fraction of images to augment (default: {IR_AUGMENT_RATIO})"
    )
    parser.add_argument(
        "--preview",
        type=int,
        nargs="?",
        const=5,
        default=None,
        help="Preview N augmented samples without modifying the dataset"
    )
    args = parser.parse_args()

    if args.preview is not None:
        preview_augmentation(num_samples=args.preview)
    else:
        run_augmentation(ratio=args.ratio)

if __name__ == "__main__":
    main()
