# Animal Guard — Farm & Wildlife Intrusion Detection

YOLO11m-based real-time detection and alert system for Indian farms. Detects 10 animal classes and triggers alerts when animals enter crop areas.

## Classes
cow, buffalo, goat, dog, wild_boar, monkey, nilgai, elephant, deer, bird

## Model Results (350 epoch baseline)

mAP@0.5: 86.3%  |  mAP@0.5:0.95: 63.7%  |  Precision: 87.8%  |  Recall: 81.1%

Trained on ~55k images (5,457 val). Best weights at `runs-10class-350/animal_guard_train/weights/best.pt` (38.6 MB). Phase 6 fine-tuning is still running on Kaggle.

## Project structure

```
├── config.py                  # hyperparams, dataset paths, class list
├── train.py                   # training script with auto-resume
├── collect_data.py            # download and merge Roboflow datasets
├── augment_nightvision.py     # synthetic IR / night-vision augmentation
├── augment_weak_classes.py    # offline augmentation for buffalo and cow
├── evaluate.py                # per-class mAP evaluation
├── detect.py                  # inference + alert pipeline
├── hard_negative_mining.py    # cow/buffalo confusion analysis
├── generate_report.py         # PDF/HTML training report
├── Colab.ipynb                # Kaggle notebook for Phase 6
├── requirements.txt
├── data/data.yaml             # YOLO dataset config
└── runs-10class-350/
    └── animal_guard_train/weights/best.pt
```

## Setup

```bash
pip install -r requirements.txt
```

## Training

We train on Kaggle T4 GPU. To run it yourself:

1. `python collect_data.py` — download and merge datasets
2. `python augment_nightvision.py` — add synthetic IR images
3. `python augment_weak_classes.py` — augment buffalo (8x) and cow (4x)
4. `python train.py` — starts training, auto-resumes from last.pt if present

## Inference

```bash
python detect.py --weights runs-10class-350/animal_guard_train/weights/best.pt --source your_video.mp4
```

## Notes

- Using YOLO11m instead of YOLO11n — the smaller model dropped elephant from 92% to 69% mAP, not worth it
- cls loss raised to 1.0 (from 0.7) to fix cow/buffalo confusion — both are large dark bovines and the model was mixing them up
- mixup reduced to 0.10 (from 0.35) — at higher values it was blending cow+buffalo images during training which made the problem worse
- buffalo gets 8x offline augmentation because it was the weakest class throughout training
- `cache=disk` is required on Kaggle otherwise the 50k image dataset OOMs the 16GB RAM limit
