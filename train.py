import argparse
import sys
from pathlib import Path

from config import (
    DATA_YAML,
    TRAIN_CONFIG,
    MERGED_DIR,
    RUNS_DIR,
    TARGET_CLASSES,
    NUM_CLASSES,
)

def verify_dataset():
    if not DATA_YAML.exists():
        print("data.yaml not found, run collect_data.py first")
        sys.exit(1)

    train_images = MERGED_DIR / "train" / "images"
    val_images = MERGED_DIR / "val" / "images"

    if not train_images.exists() or not val_images.exists():
        print("dataset directories missing, run collect_data.py first")
        sys.exit(1)

    train_count = len(list(train_images.glob("*")))
    val_count = len(list(val_images.glob("*")))

    if train_count == 0:
        print("no training images found, run collect_data.py first")
        sys.exit(1)

    print(f"dataset ok: {train_count} train images, {val_count} val images")
    print(f"classes ({NUM_CLASSES}): {', '.join(TARGET_CLASSES)}")

    if val_count == 0:
        print("WARNING: no val images found, evaluation won't work properly")

    return train_count, val_count

def train(args):
    """Run YOLO11m training with configured hyperparameters."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] ultralytics not found. Installing...")
        import os
        os.system(f"{sys.executable} -m pip install ultralytics")
        from ultralytics import YOLO

    train_count, val_count = verify_dataset()

    config = TRAIN_CONFIG.copy()

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.model is not None:
        config["model"] = args.model
    if args.workers is not None:
        config["workers"] = args.workers
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.device is not None:
        config["device"] = args.device

    model_name = config.pop("model")

    print(f"model: {model_name}, epochs: {config['epochs']}, imgsz: {config['imgsz']}, batch: {config['batch']}")
    print(f"output: {config['project']}/{config['name']}")

    if args.resume:
        last_ckpt = Path(config["project"]) / config["name"] / "weights" / "last.pt"
        if last_ckpt.exists():
            import torch
            ckpt = torch.load(str(last_ckpt), map_location="cpu", weights_only=False)
            has_optimizer = ckpt.get("optimizer") is not None
            completed_epoch = ckpt.get("epoch", -1)
            total_epochs_target = config["epochs"]

            m = ckpt.get("ema") or ckpt.get("model")
            checkpoint_classes = getattr(m, "names", {}) if m else {}
            class_count_mismatch = len(checkpoint_classes) != NUM_CLASSES

            if class_count_mismatch:
                # class count changed, can't resume directly - use as pretrained weights
                print(f"class count changed ({len(checkpoint_classes)} -> {NUM_CLASSES}), doing transfer learning")
                model = YOLO(str(last_ckpt))
                results = model.train(data=str(DATA_YAML), **config)

            elif has_optimizer and completed_epoch >= 0:
                # mid-run checkpoint, optimizer is still in the file so YOLO can resume exactly
                print(f"resuming from epoch {completed_epoch + 1}: {last_ckpt}")
                model = YOLO(str(last_ckpt))
                results = model.train(data=str(DATA_YAML), resume=True, **config)

            else:
                # training finished cleanly - optimizer stripped, so we continue manually
                # YOLO sets epoch=-1 when done. CHECKPOINT_EPOCH tells us how many epochs
                # are baked into this checkpoint so we can calculate remaining epochs.
                from config import CHECKPOINT_EPOCH
                prev_epochs = CHECKPOINT_EPOCH
                remaining = max(total_epochs_target - int(prev_epochs), 1)
                config["epochs"] = remaining
                print(f"continuing from {prev_epochs} epochs, training {remaining} more to reach {total_epochs_target}")
                model = YOLO(str(last_ckpt))
                results = model.train(data=str(DATA_YAML), **config)

        else:
            print(f"checkpoint not found at {last_ckpt}, starting fresh with {model_name}")
            model = YOLO(model_name)
            results = model.train(data=str(DATA_YAML), **config)
    else:
        print(f"fresh training with {model_name}")
        model = YOLO(model_name)
        results = model.train(data=str(DATA_YAML), **config)

    best_weights = Path(config["project"]) / config["name"] / "weights" / "best.pt"
    last_weights = Path(config["project"]) / config["name"] / "weights" / "last.pt"

    if best_weights.exists():
        size_mb = best_weights.stat().st_size / (1024 * 1024)
        print(f"best.pt: {best_weights} ({size_mb:.1f} MB)")
    if last_weights.exists():
        print(f"last.pt: {last_weights}")

    print(f"run evaluate.py --weights {best_weights} to check performance")

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Animal Guard — Train YOLO11n for Indian farm animal detection"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help=f"Number of training epochs (default: {TRAIN_CONFIG['epochs']})"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help=f"Batch size, -1 for auto (default: {TRAIN_CONFIG['batch']})"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"Base model name (default: {TRAIN_CONFIG['model']})"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"DataLoader workers (default: {TRAIN_CONFIG['workers']})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help=f"Training image size (default: {TRAIN_CONFIG['imgsz']})"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 0 for GPU, cpu for CPU"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint"
    )
    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
