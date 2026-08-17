# Animal Guard — Farm & Wildlife Intrusion Detection

YOLO11m-based real-time detection system for Indian farms. Detects 10 animal classes and can trigger alerts when animals enter crop areas.

## Classes

cow, buffalo, goat, dog, wild_boar, monkey, nilgai, elephant, deer, bird

## Results (350 epoch baseline)

| Metric | Score |
|:---|:---|
| mAP@0.5 | 86.3% |
| mAP@0.5:0.95 | 63.7% |
| Precision | 87.8% |
| Recall | 81.1% |

Trained on ~55k images (5,457 val images). Phase 6 fine-tuning is still running on Kaggle.

## Project structure

```
├── config.py                  # hyperparams, dataset paths, class list
├── train.py                   # training with auto-resume from checkpoint
├── collect_data.py            # download and merge Roboflow datasets
├── augment_nightvision.py     # synthetic IR / night-vision augmentation
├── augment_weak_classes.py    # offline augmentation for buffalo (8x) and cow (4x)
├── evaluate.py                # per-class mAP evaluation
├── detect.py                  # inference + alert pipeline
├── hard_negative_mining.py    # cow/buffalo confusion analysis
├── generate_report.py         # PDF training report generator
├── requirements.txt
├── data/data.yaml             # YOLO dataset config (class names + split paths)
└── runs-10class-350/
    └── animal_guard_train/
        ├── weights/best.pt    # 350-epoch trained model (38.6 MB, via Git LFS)
        └── *.png              # training charts and validation predictions
```

## Setup

```bash
git clone https://github.com/KRISH71819/animal-detection.git
cd animal-detection
git lfs pull            # downloads best.pt
pip install -r requirements.txt
```

## Running inference (no dataset needed)

```bash
python detect.py --weights runs-10class-350/animal_guard_train/weights/best.pt --source your_video.mp4
```

## Running evaluation

Evaluation needs the validation dataset. Run collect_data.py first to download it.

**1. Set your Roboflow API key**

Get your key from https://app.roboflow.com → Settings → API Keys

```bash
cp .env.example .env
# open .env and replace 'your_roboflow_api_key_here' with your actual key
```

**2. Download datasets (~20-30 min)**

```bash
python collect_data.py
```

**3. Run evaluation**

```bash
# GPU
python evaluate.py --weights runs-10class-350/animal_guard_train/weights/best.pt --device 0

# CPU
python evaluate.py --weights runs-10class-350/animal_guard_train/weights/best.pt --device cpu
```

## Training from scratch

```bash
python collect_data.py           # download datasets (needs ROBOFLOW_API_KEY)
python augment_nightvision.py    # add synthetic IR images
python augment_weak_classes.py   # augment buffalo (8x) and cow (4x)
python train.py                  # start training
python train.py --resume         # resume from last checkpoint
```
