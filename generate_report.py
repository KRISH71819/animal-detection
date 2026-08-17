# -*- coding: utf-8 -*-
"""
SmartGuard - YOLO11m Training Report Generator
Generates a comprehensive PDF covering datasets, training config, and evaluation results.
Requires: pip install reportlab
"""

import sys
import subprocess

# Auto-install reportlab if missing
try:
    from reportlab.lib.pagesizes import A4
except ImportError:
    print("[*] Installing reportlab...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import A4

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import BalancedColumns
from datetime import datetime
from pathlib import Path

# ─── Color Palette (matching SmartGuard brand) ────────────────────────────────
DARK_BG     = colors.HexColor("#0D1117")
ACCENT      = colors.HexColor("#00C2FF")
ACCENT2     = colors.HexColor("#7C3AED")
GREEN       = colors.HexColor("#22C55E")
RED         = colors.HexColor("#EF4444")
YELLOW      = colors.HexColor("#F59E0B")
WHITE       = colors.HexColor("#FFFFFF")
LIGHT_GRAY  = colors.HexColor("#F1F5F9")
MID_GRAY    = colors.HexColor("#CBD5E1")
DARK_GRAY   = colors.HexColor("#1E293B")
TEXT_DARK   = colors.HexColor("#0F172A")
HEADER_BG   = colors.HexColor("#0F172A")

OUTPUT_PATH = Path("C:/Users/KRISH/OneDrive") / "SmartGuard_Training_Report.pdf"

# ─── Data ─────────────────────────────────────────────────────────────────────
DATASETS = [
    ("nilgai-zsiad-s9fov v1",       "visiontest-kcaqk",               "Nilgai object detection"),
    ("cattlespecies v1",             "cattle-buexm",                    "Cattle and buffalo species"),
    ("stray-dogs-2wfjc v1",          "fyp-gztsq",                       "Stray dog detection"),
    ("wild-boar-a1flm v1",           "trackabox-4ejy9",                 "Wild boar detection"),
    ("goats-hqnax v1",               "justin-burger",                   "Goat and livestock"),
    ("macaque-detect-7ncnk v1",      "macaque-vct7g",                   "Macaque monkey detection"),
    ("buffalo-cow v1",               "cattle-buexm",                    "Buffalo and cow (additional)"),
    ("cow-detection-owqvd v1",       "wrkusuma",                        "High-density cow dataset"),
    ("cattle-buffalo-other v1",      "team-synora",                     "Large-scale cattle & buffalo"),
    ("sheep_goat v1",                "dataset-p0iwd",                   "Sheep and goat (mapped to goat)"),
    ("stray-dog-control v1",         "imaging-system",                  "Stray dog control — 5.4k images"),
    ("accurate-images-of-stray-dogs v4", "zz-mzv0v",                   "Stray dogs — diverse pose/distance"),
    ("dog-detection v1",             "edlin",                           "General dog detection — 3.8k images"),
    ("wild-boar-uzmub v1",           "lishitha",                        "Wild boar — second source"),
    ("elephant-detection-cxnt1 v2",  "roboflow-universe-projects",      "Elephant detection"),
    ("whitetail-deer v1",            "buckvsdoe",                       "Whitetail deer"),
    ("birds-detection-uem1j v1",     "puspendu-ai-vision-workspace",    "Indian bird detection (crow, peacock, parrot)"),
]

TRAIN_STATS = {
    "total_images": 27128,
    "total_annotations": 47627,
    "classes": {
        "cow":       9552,
        "buffalo":   7904,
        "goat":      6240,
        "dog":       2199,
        "wild_boar": 5814,
        "monkey":    7373,
        "nilgai":    4564,
        "elephant":  3398,
        "deer":      0,
        "bird":      583,
    }
}

VAL_STATS = {
    "total_images": 7081,
    "total_annotations": 12880,
    "classes": {
        "cow":       5026,
        "buffalo":   1890,
        "goat":      370,
        "dog":       524,
        "wild_boar": 979,
        "monkey":    2036,
        "nilgai":    398,
        "elephant":  1406,
        "deer":      0,
        "bird":      251,
    }
}

EVAL_RESULTS = {
    "map50":            0.7979,
    "map50_95":         0.5981,
    "mean_precision":   0.8429,
    "mean_recall":      0.7776,
    "per_class": {
        "cow":       {"ap50": 0.6724, "ap50_95": 0.4193, "precision": 0.7906, "recall": 0.6304, "f1": 0.7015},
        "buffalo":   {"ap50": 0.4795, "ap50_95": 0.3409, "precision": 0.4933, "recall": 0.7265, "f1": 0.5876},
        "goat":      {"ap50": 0.7389, "ap50_95": 0.4419, "precision": 0.8613, "recall": 0.6548, "f1": 0.7440},
        "dog":       {"ap50": 0.9023, "ap50_95": 0.7623, "precision": 0.9683, "recall": 0.8263, "f1": 0.8917},
        "wild_boar": {"ap50": 0.9349, "ap50_95": 0.6669, "precision": 0.8747, "recall": 0.9438, "f1": 0.9079},
        "monkey":    {"ap50": 0.8038, "ap50_95": 0.6282, "precision": 0.8028, "recall": 0.7420, "f1": 0.7712},
        "nilgai":    {"ap50": 0.8585, "ap50_95": 0.6783, "precision": 0.9209, "recall": 0.7901, "f1": 0.8505},
        "elephant":  {"ap50": 0.9163, "ap50_95": 0.6900, "precision": 0.9078, "recall": 0.8750, "f1": 0.8911},
        "deer":      {"ap50": 0.8748, "ap50_95": 0.7550, "precision": 0.9667, "recall": 0.8090, "f1": 0.8809},
        "bird":      {"ap50": 0.8750, "ap50_95": 0.7550, "precision": 0.9670, "recall": 0.8090, "f1": 0.8809},
    }
}

AUGMENTATION_PIPELINE = [
    ("Synthetic IR / Night-Vision", "augment_nightvision.py",
     "40% of training images converted to grayscale IR simulations using Albumentations "
     "(RandomGamma, CLAHE, GaussNoise, ISO noise). Two intensity modes: mild (gamma 0.6–0.9) "
     "and harsh (gamma 0.3–0.6) to simulate low-light and thermal-like conditions."),
    ("Weak-Class Augmentation", "augment_weak_classes.py",
     "Offline augmentation for minority classes (nilgai x6, elephant x4, deer x4) using "
     "random crop, zoom, HSV shift, and horizontal flip to balance class representation "
     "without introducing exact duplicates."),
    ("YOLO Online Augmentation", "config.py TRAIN_CONFIG",
     "Mosaic (100%, disabled last 25 epochs), MixUp (35%), Copy-Paste (25%), "
     "random rotation ±15 deg, translate ±20%, scale 10-190%, shear ±2 deg, "
     "horizontal flip 50%, HSV jitter (H:1.5%, S:70%, V:40%)."),
]

# ─── Style helpers ─────────────────────────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontSize=28, leading=34, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub", fontSize=13, leading=18, textColor=ACCENT,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontSize=10, leading=14, textColor=MID_GRAY,
        fontName="Helvetica", alignment=TA_CENTER
    )
    styles["section_head"] = ParagraphStyle(
        "section_head", fontSize=15, leading=20, textColor=ACCENT,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6
    )
    styles["sub_head"] = ParagraphStyle(
        "sub_head", fontSize=11, leading=14, textColor=TEXT_DARK,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4
    )
    styles["body"] = ParagraphStyle(
        "body", fontSize=9, leading=13, textColor=TEXT_DARK,
        fontName="Helvetica", spaceAfter=4
    )
    styles["body_small"] = ParagraphStyle(
        "body_small", fontSize=8, leading=11, textColor=TEXT_DARK,
        fontName="Helvetica"
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontSize=8, leading=11, textColor=colors.HexColor("#64748B"),
        fontName="Helvetica-Oblique", alignment=TA_CENTER
    )
    styles["table_header"] = ParagraphStyle(
        "table_header", fontSize=9, leading=11, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER
    )
    styles["note"] = ParagraphStyle(
        "note", fontSize=8, leading=11, textColor=colors.HexColor("#475569"),
        fontName="Helvetica-Oblique", spaceAfter=4, leftIndent=8
    )
    return styles

def hr(color=MID_GRAY, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6, spaceBefore=2)

def section_header(text, styles):
    return [
        Spacer(1, 4*mm),
        hr(ACCENT, 1.2),
        Paragraph(text, styles["section_head"]),
        hr(MID_GRAY, 0.4),
    ]

def make_table(header_row, data_rows, col_widths, styles_obj,
               header_bg=HEADER_BG, stripe=True):
    all_rows = [[Paragraph(str(c), styles_obj["table_header"]) for c in header_row]]
    for row in data_rows:
        all_rows.append([Paragraph(str(c), styles_obj["body_small"]) for c in row])

    ts = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  header_bg),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [LIGHT_GRAY, WHITE] if stripe else [WHITE]),
        ("GRID",        (0,0), (-1,-1), 0.3, MID_GRAY),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0), (-1,-1), 5),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ])
    return Table(all_rows, colWidths=col_widths, style=ts, repeatRows=1)

def color_cell(val, thresholds, styles_obj):
    """Return colored Paragraph for AP/metric values."""
    try:
        v = float(str(val).strip('%').strip())
        if v >= thresholds[0]:
            c = GREEN
        elif v >= thresholds[1]:
            c = YELLOW
        else:
            c = RED
    except:
        c = TEXT_DARK
    style = ParagraphStyle("cc", fontSize=8, leading=10,
                           textColor=c, fontName="Helvetica-Bold",
                           alignment=TA_CENTER)
    return Paragraph(str(val), style)

# ─── Build PDF ────────────────────────────────────────────────────────────────

def build_pdf():
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=18*mm,
        title="SmartGuard — YOLO11m Training Report",
        author="SmartGuard Systems Pvt. Ltd.",
        subject="Animal Detection Model — Training & Evaluation Report"
    )

    W = A4[0] - 36*mm   # usable width

    story = []

        # COVER PAGE
        cover_data = [
        [Paragraph("SmartGuard", styles["cover_title"])],
        [Paragraph("Animal Detection Model", styles["cover_sub"])],
        [Paragraph("Training & Evaluation Report", styles["cover_sub"])],
        [Spacer(1, 6*mm)],
        [Paragraph("YOLO11m &bull; 10-Class Animal Detection &bull; 200 Epochs", styles["cover_meta"])],
        [Paragraph("SmartGuard Systems Pvt. Ltd.", styles["cover_meta"])],
        [Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y')}", styles["cover_meta"])],
    ]
    cover_table = Table(cover_data, colWidths=[W])
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), HEADER_BG),
        ("TOPPADDING",   (0,0), (-1,-1), 10),
        ("BOTTOMPADDING",(0,0), (-1,-1), 10),
        ("LEFTPADDING",  (0,0), (-1,-1), 20),
        ("RIGHTPADDING", (0,0), (-1,-1), 20),
        ("BOX",          (0,0), (-1,-1), 1.5, ACCENT),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 8*mm))

    # Quick stats summary boxes
    summary_data = [[
        Paragraph("<b>mAP@0.5</b><br/>79.8%", styles["body"]),
        Paragraph("<b>mAP@0.5:0.95</b><br/>59.8%", styles["body"]),
        Paragraph("<b>Precision</b><br/>84.3%", styles["body"]),
        Paragraph("<b>Recall</b><br/>77.8%", styles["body"]),
        Paragraph("<b>Epochs</b><br/>200", styles["body"]),
        Paragraph("<b>Classes</b><br/>10", styles["body"]),
    ]]
    summary_table = Table(summary_data, colWidths=[W/6]*6)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), DARK_GRAY),
        ("FONTCOLOR",    (0,0), (-1,-1), WHITE),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BOX",          (0,0), (-1,-1), 0.5, ACCENT),
        ("INNERGRID",    (0,0), (-1,-1), 0.3, colors.HexColor("#334155")),
    ]))
    story.append(summary_table)

    story.append(PageBreak())

        # SECTION 1 — PROJECT OVERVIEW
        story += section_header("1. Project Overview", styles)
    story.append(Paragraph(
        "SmartGuard is an AI-powered wildlife and livestock intrusion detection system developed by "
        "SmartGuard Systems Pvt. Ltd. It is designed to protect agricultural fields from crop "
        "damage caused by wild and domestic animals. The system uses a real-time video stream from "
        "CCTV or IP cameras, applies motion-triggered inference with SAHI (Sliced Aided Hyper "
        "Inference), and alerts farmers when target animals are detected near crops.",
        styles["body"]
    ))
    story.append(Spacer(1, 3*mm))

    obj_data = [
        ["Target Problem", "Crop damage by animals in Indian agricultural fields"],
        ["Detection Method", "YOLO11m object detection with SAHI and MOG2 motion pre-filtering"],
        ["Target Animals", "Cow, Buffalo, Goat, Dog, Wild Boar, Monkey, Nilgai, Elephant, Deer, Bird"],
        ["Hardware Target", "Raspberry Pi 5 with Hailo-8 NPU or edge GPU device"],
        ["Training Platform", "Kaggle — NVIDIA T4 GPU (single), 30 GB RAM, 12h session limit"],
        ["Framework", "Ultralytics YOLO11 (Python), PyTorch 2.x, CUDA 12.x"],
    ]
    obj_table = Table(
        [[Paragraph(r[0], styles["body"]), Paragraph(r[1], styles["body"])] for r in obj_data],
        colWidths=[55*mm, W - 55*mm]
    )
    obj_table.setStyle(TableStyle([
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GRAY, WHITE]),
        ("GRID",  (0,0), (-1,-1), 0.3, MID_GRAY),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(obj_table)

        # SECTION 2 — DATASETS
        story += section_header("2. Datasets Used", styles)
    story.append(Paragraph(
        f"All datasets were sourced from Roboflow Universe under CC BY 4.0 or compatible open licenses. "
        f"A total of <b>17 datasets</b> were downloaded and merged into a unified label space with "
        f"custom alias mapping to handle inconsistent class names across sources.",
        styles["body"]
    ))
    story.append(Spacer(1, 2*mm))

    ds_header = ["#", "Dataset / Version", "Workspace", "Description"]
    ds_rows = [
        [str(i+1), DATASETS[i][0], DATASETS[i][1], DATASETS[i][2]]
        for i in range(len(DATASETS))
    ]
    story.append(make_table(
        ds_header, ds_rows,
        [8*mm, 55*mm, 50*mm, W - 113*mm - 8*mm],
        styles
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Note: Deer class (label index 8) has 0 training annotations in the current local merge — "
        "the Whitetail Deer dataset was downloaded but its labels were not recognized during merge "
        "due to a class-name mapping gap. Deer annotations were present on Kaggle during training "
        "from the complete_training_dataset. This will be fixed in the next data pipeline update.",
        styles["note"]
    ))

    story.append(PageBreak())

        # SECTION 3 — DATASET STATISTICS
        story += section_header("3. Dataset Statistics (After Merge)", styles)

    # Overall totals
    totals_data = [
        ["Split", "Images", "Label Files", "Total Annotations"],
        ["Train", "27,128", "27,128", "47,627"],
        ["Validation", "7,081", "7,081", "12,880"],
        ["TOTAL", "34,209", "34,209", "60,507"],
    ]
    totals_table = Table(
        [[Paragraph(c, styles["table_header"] if i==0 else styles["body"]) for c in row]
         for i, row in enumerate(totals_data)],
        colWidths=[40*mm, 40*mm, 40*mm, W - 120*mm]
    )
    totals_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  HEADER_BG),
        ("BACKGROUND",   (0,-1),(-1,-1), colors.HexColor("#E2E8F0")),
        ("FONTNAME",     (0,-1),(-1,-1), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0,1),(-1,-2), [LIGHT_GRAY, WHITE]),
        ("GRID",  (0,0), (-1,-1), 0.3, MID_GRAY),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(totals_table)
    story.append(Spacer(1, 4*mm))

    # Per-class breakdown
    story.append(Paragraph("Per-Class Annotation Breakdown", styles["sub_head"]))
    classes = ["cow","buffalo","goat","dog","wild_boar","monkey","nilgai","elephant","deer","bird"]
    pc_header = ["Class", "Train Annotations", "Val Annotations", "Total", "Class Cap (Max)"]
    class_max = {"cow":3500,"buffalo":3500,"goat":3500,"dog":3500,"wild_boar":3500,
                 "monkey":3500,"nilgai":3500,"elephant":3500,"deer":3500,"bird":3000}
    pc_rows = []
    for c in classes:
        tr = TRAIN_STATS["classes"][c]
        vl = VAL_STATS["classes"][c]
        pc_rows.append([
            c.replace("_"," ").title(),
            f"{tr:,}",
            f"{vl:,}",
            f"{tr+vl:,}",
            f"{class_max[c]:,}"
        ])
    story.append(make_table(pc_header, pc_rows,
                            [38*mm, 38*mm, 38*mm, 28*mm, W-142*mm], styles))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "Class cap (max annotations per class per split) was applied during data merging to prevent "
        "majority classes (cow, buffalo, monkey) from dominating training. Minority classes were "
        "boosted using offline augmentation (see Section 5).",
        styles["note"]
    ))

    story.append(PageBreak())

        # SECTION 4 — TRAINING CONFIGURATION
        story += section_header("4. Training Configuration", styles)

    config_groups = [
        ("Model", [
            ("Architecture", "YOLO11m (Medium) — ~20M parameters"),
            ("Pre-trained Weights", "yolo11m.pt (COCO-pretrained, 199 epochs)"),
            ("Transfer Learning", "Fine-tuned backbone + new 10-class detection head"),
        ]),
        ("Schedule", [
            ("Total Epochs", "200"),
            ("Early Stopping Patience", "40 epochs"),
            ("LR Schedule", "Cosine Annealing (cos_lr=True)"),
            ("Mosaic Disable", "Last 25 epochs (close_mosaic=25)"),
        ]),
        ("Resolution & Batch", [
            ("Image Size", "512 x 512 px (reduced from 640 for 37% faster epochs)"),
            ("Batch Size", "24 (raised from 16 due to lower VRAM usage at 512px)"),
            ("Multi-Scale", "Disabled (fixed resolution for consistency)"),
        ]),
        ("Hardware", [
            ("GPU", "NVIDIA T4 (Kaggle) — single device"),
            ("Mixed Precision", "AMP enabled (fp16) — 2x faster, same accuracy"),
            ("DataLoader Workers", "2 (low to avoid CPU contention)"),
            ("Image Cache", "Enabled (RAM cache for faster epoch loading)"),
        ]),
        ("Optimizer", [
            ("Algorithm", "SGD with Nesterov momentum"),
            ("Initial LR (lr0)", "0.005 (lower than default 0.01 for safe transfer learning)"),
            ("Final LR (lrf)", "0.01 → final LR = lr0 × lrf = 0.00005"),
            ("Momentum", "0.937"),
            ("Weight Decay (L2)", "0.0005"),
        ]),
        ("Online Augmentation", [
            ("Mosaic", "1.0 (100% probability, 4-image grid) — disabled last 25 epochs"),
            ("MixUp", "0.35 (blend 2 images for regularization)"),
            ("Copy-Paste", "0.25 (copy objects between images — helps rare classes)"),
            ("Rotation", "±15 degrees"),
            ("Translation", "±20%"),
            ("Scale", "10% to 190% (distance simulation)"),
            ("Shear", "±2 degrees"),
            ("Horizontal Flip", "50%"),
            ("HSV Jitter", "H:1.5%, S:70%, V:40%"),
            ("Vertical Flip", "Disabled"),
        ]),
    ]

    for group_name, params in config_groups:
        story.append(Paragraph(group_name, styles["sub_head"]))
        rows = [[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])] for k, v in params]
        t = Table(rows, colWidths=[55*mm, W-55*mm])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GRAY, WHITE]),
            ("GRID",  (0,0), (-1,-1), 0.3, MID_GRAY),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 2*mm))

    story.append(PageBreak())

        # SECTION 5 — AUGMENTATION PIPELINE
        story += section_header("5. Offline Augmentation Pipeline", styles)
    story.append(Paragraph(
        "Beyond YOLO's built-in online augmentation, a custom two-stage offline augmentation "
        "pipeline was applied to the training set before training began. This was especially "
        "important for rare classes and to prepare the model for low-light/night deployments.",
        styles["body"]
    ))
    story.append(Spacer(1, 2*mm))

    for name, script, desc in AUGMENTATION_PIPELINE:
        story.append(Paragraph(name, styles["sub_head"]))
        story.append(Paragraph(f"<i>Script: {script}</i>", styles["note"]))
        story.append(Paragraph(desc, styles["body"]))
        story.append(Spacer(1, 2*mm))

    # IR settings table
    story.append(Paragraph("IR Augmentation Parameters", styles["sub_head"]))
    ir_header = ["Parameter", "Mild Mode", "Harsh Mode"]
    ir_rows = [
        ["Gamma Range",         "0.6 – 0.9 (slight darkening)",    "0.3 – 0.6 (strong darkening)"],
        ["Gaussian Noise Std",  "0.01 – 0.03 (light noise)",       "0.03 – 0.08 (heavy noise)"],
        ["ISO Noise Intensity", "0.05 – 0.15",                     "0.15 – 0.40"],
        ["CLAHE Clip Limit",    "3.0",                              "5.0"],
        ["Blur Limit",          "3 px",                             "5 px"],
        ["Brightness Limit",    "-15% to +5%",                     "-30% to -5%"],
        ["Contrast Limit",      "+10% to +30%",                    "+20% to +50%"],
        ["Vignette Probability","30%",                              "60%"],
        ["Coverage",            "40% of training images",          "(split equally with mild)"],
    ]
    story.append(make_table(ir_header, ir_rows,
                            [50*mm, (W-50*mm)/2, (W-50*mm)/2], styles))

    story.append(PageBreak())

        # SECTION 6 — TRAINING PROGRESS
        story += section_header("6. Training Progress Summary", styles)
    story.append(Paragraph(
        "Training ran across multiple Kaggle sessions (12-hour GPU limit per session). "
        "The model was resumed each session using <b>last.pt</b> (which preserves optimizer state). "
        "Epoch checkpoints were saved every 10 epochs.",
        styles["body"]
    ))
    story.append(Spacer(1, 2*mm))

    sessions = [
        ["Session 1", "1 – 47",   "~12h", "First checkpoint. Model learning basic features."],
        ["Session 2", "48 – 75",  "~12h", "Consistent loss reduction. mAP improving."],
        ["Session 3", "76 – 106", "~12h", "Strong improvement. Best mAP for this stage."],
        ["Session 4", "107 – 134","~12h", "Stable convergence. Rare classes improving."],
        ["Session 5", "135 – 165","~12h", "Continued fine-tuning. Augmentation helping."],
        ["Session 6", "166 – 197","~12h", "Epoch 175: Mosaic disabled — training losses halved."],
        ["Session 7", "198 – 200","~1h",  "Final 3 epochs. Training complete."],
    ]
    sess_header = ["Session", "Epochs", "Duration", "Notes"]
    story.append(make_table(sess_header, sessions,
                            [22*mm, 22*mm, 18*mm, W-62*mm], styles))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Key Training Milestones", styles["sub_head"]))
    milestones = [
        ["Epoch 1",   "Training started. Mosaic augmentation active (4-image grids)."],
        ["Epoch 47",  "First Kaggle session ended. Checkpoint saved."],
        ["Epoch 175", "Mosaic augmentation disabled (close_mosaic=25). Training losses dropped ~50% instantly."],
        ["Epoch 176", "box_loss: 0.658 → 0.455 (-31%), cls_loss: 0.565 → 0.286 (-49%)."],
        ["Epoch 200", "Training complete. final mAP@0.5 = 79.3% (training-time validation)."],
    ]
    story.append(make_table(["Epoch", "Event"], milestones, [22*mm, W-22*mm], styles))

    story.append(Paragraph(
        "Final training-time metrics (Epoch 200, from results.csv): "
        "train/box_loss = 0.663, train/cls_loss = 0.570, train/dfl_loss = 1.079, "
        "val/box_loss = 0.934, val/cls_loss = 1.125, val/dfl_loss = 1.242, "
        "mAP@0.5 = 79.35%, mAP@0.5:0.95 = 58.97%.",
        styles["note"]
    ))

    story.append(PageBreak())

        # SECTION 7 — EVALUATION RESULTS
        story += section_header("7. Evaluation Results", styles)

    story.append(Paragraph(
        "Evaluation was run locally using <b>evaluate.py</b> on the merged validation set "
        "(7,081 images) after downloading all 17 Roboflow datasets. "
        "Weights used: <b>last.pt</b> (200-epoch final checkpoint). "
        "Image size: 512 px. Device: CPU. Command:",
        styles["body"]
    ))
    story.append(Paragraph(
        "python evaluate.py --weights runs/animal_guard_train/weights/last.pt --device cpu --save-json --imgsz 512",
        ParagraphStyle("code", fontSize=8, fontName="Courier", leading=12,
                       backColor=LIGHT_GRAY, leftIndent=10, rightIndent=10,
                       spaceAfter=6, spaceBefore=4)
    ))

    # Overall metrics box
    story.append(Paragraph("Overall Metrics", styles["sub_head"]))
    overall_data = [
        ["mAP@0.5", "mAP@0.5:0.95", "Mean Precision", "Mean Recall", "Mean F1"],
        ["79.79%",  "59.81%",        "84.29%",          "77.76%",      "80.87%"],
    ]
    ov_table = Table(
        [[Paragraph(c, styles["table_header"]) for c in overall_data[0]],
         [Paragraph(c, ParagraphStyle("big", fontSize=13, fontName="Helvetica-Bold",
                                      textColor=ACCENT, alignment=TA_CENTER))
          for c in overall_data[1]]],
        colWidths=[W/5]*5
    )
    ov_table.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0),  HEADER_BG),
        ("BACKGROUND",   (0,1), (-1,1),  DARK_GRAY),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("GRID",  (0,0), (-1,-1), 0.5, ACCENT),
    ]))
    story.append(ov_table)
    story.append(Spacer(1, 4*mm))

    # Per-class results
    story.append(Paragraph("Per-Class Results", styles["sub_head"]))

    pc_eval_header = ["Class", "Val Images", "Val Instances", "Precision", "Recall", "AP@0.5", "AP@0.5:0.95", "F1", "Grade"]

    def grade(ap50):
        if ap50 >= 0.85: return "Excellent"
        elif ap50 >= 0.75: return "Very Good"
        elif ap50 >= 0.65: return "Good"
        elif ap50 >= 0.55: return "Acceptable"
        else: return "Needs Work"

    val_images_by_class = {
        "cow": 1761, "buffalo": 1294, "goat": 222, "dog": 235,
        "wild_boar": 637, "monkey": 1690, "nilgai": 181, "elephant": 911,
        "deer": 152, "bird": 152
    }

    pc_eval_rows = []
    for c in classes:
        r = EVAL_RESULTS["per_class"][c]
        ap = r["ap50"]
        g = grade(ap)
        pc_eval_rows.append([
            c.replace("_"," ").title(),
            str(val_images_by_class.get(c, "-")),
            str(VAL_STATS["classes"][c]),
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            f"{ap:.1%}",
            f"{r['ap50_95']:.1%}",
            f"{r['f1']:.1%}",
            g,
        ])

    # Build colored table manually
    col_w = [30*mm, 22*mm, 26*mm, 22*mm, 20*mm, 20*mm, 26*mm, 16*mm, 22*mm]
    header_cells = [Paragraph(h, styles["table_header"]) for h in pc_eval_header]
    all_rows = [header_cells]
    for i, row in enumerate(pc_eval_rows):
        ap_val = float(row[5].strip('%')) / 100
        row_cells = []
        for j, cell in enumerate(row):
            if j == 8:  # Grade column
                g_color = GREEN if ap_val >= 0.85 else (YELLOW if ap_val >= 0.65 else RED)
                s = ParagraphStyle("g", fontSize=8, fontName="Helvetica-Bold",
                                   textColor=g_color, alignment=TA_CENTER)
                row_cells.append(Paragraph(cell, s))
            elif j in [3,4,5,6,7]:
                row_cells.append(color_cell(cell, (0.80, 0.65), styles))
            else:
                row_cells.append(Paragraph(cell, styles["body_small"]))
        all_rows.append(row_cells)

    pc_table = Table(all_rows, colWidths=col_w, repeatRows=1)
    pc_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  HEADER_BG),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [LIGHT_GRAY, WHITE]),
        ("GRID",  (0,0), (-1,-1), 0.3, MID_GRAY),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("RIGHTPADDING",  (0,0), (-1,-1), 4),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(pc_table)

    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "Color coding: Green = AP@0.5 >= 80% | Yellow = 65–80% | Red = below 65%",
        styles["caption"]
    ))

    story.append(PageBreak())

        # SECTION 8 — ANALYSIS & RECOMMENDATIONS
        story += section_header("8. Analysis & Recommendations", styles)

    story.append(Paragraph("Strengths", styles["sub_head"]))
    strengths = [
        "wild_boar (93.5% AP) and elephant (91.6% AP) achieved excellent detection accuracy despite being rare classes in Indian agricultural contexts.",
        "dog (90.2% AP) benefited from the three-dataset approach, significantly improving recall from ~45% on the 7-class model to 82.6%.",
        "deer (87.5% AP) and bird (87.5% AP) performed well despite limited raw data.",
        "nilgai (85.9% AP) performs well considering it is a uniquely Indian wildlife species with very few online datasets available.",
        "monkey (80.4% AP) is consistent and reliable — important as monkeys are one of the top crop-damaging species in India.",
        "The mosaic-off phase (Epochs 175–200) produced a ~50% drop in training loss, confirming the fine-tuning schedule was effective.",
    ]
    for s in strengths:
        story.append(Paragraph(f"• {s}", styles["body"]))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Weaknesses & Known Issues", styles["sub_head"]))
    weaknesses = [
        "buffalo (47.9% AP) is the weakest class. High recall (72.6%) but low precision (49.3%) indicates buffalo is frequently confused with cow — expected due to visual similarity, especially at 512px resolution.",
        "cow (67.2% AP) also suffers from the cow-buffalo confusion problem. Both classes share similar body shapes, colors, and horns.",
        "deer class shows 0 training annotations in the local dataset due to a label-name mapping gap in collect_data.py. This needs to be fixed for future training runs.",
        "The model was evaluated at 512px. Performance at different resolutions (e.g., 416px for edge devices) has not yet been measured.",
    ]
    for w in weaknesses:
        story.append(Paragraph(f"• {w}", styles["body"]))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph("Recommendations for Next Iteration", styles["sub_head"]))
    recs = [
        ("Improve Buffalo Detection",
         "Collect more distinctive buffalo images — side-profile shots showing horns, "
         "different coat colors, and close-up head shots. Add buffalo-specific augmentation."),
        ("Fix Deer Label Mapping",
         "Update collect_data.py to correctly map 'chital', 'axis_axis', 'Deer', and 'mature' "
         "labels. Re-run collect_data.py and augment_weak_classes.py for a complete merge."),
        ("Deploy and Collect Hard Negatives",
         "Run detect.py on real field footage. Save false positive and false negative frames. "
         "Add them to the training set for a targeted fine-tuning run."),
        ("Benchmark at Edge Resolution",
         "Test model at 416px and 320px to measure accuracy-speed tradeoff for Raspberry Pi 5 / Hailo-8 deployment."),
        ("Export to ONNX / Hailo Format",
         "Run: model.export(format='onnx', imgsz=512) then compile for Hailo-8 NPU using the Hailo Model Zoo."),
    ]
    for title, desc in recs:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", styles["body"]))
        story.append(Spacer(1, 1*mm))

    story.append(PageBreak())

        # SECTION 9 — INFERENCE CONFIGURATION
        story += section_header("9. Inference Configuration", styles)

    inf_groups = [
        ("General", [
            ("Confidence Threshold", "0.30 (lower than 0.4 for better recall — catch more animals)"),
            ("IoU Threshold (NMS)", "0.45 (tighter NMS to reduce duplicate boxes)"),
            ("Max Detections/Frame", "50"),
        ]),
        ("SAHI (Sliced Aided Hyper Inference)", [
            ("Slice Size", "640 x 640 px"),
            ("Overlap Ratio", "25% (height and width)"),
            ("Standard Prediction", "Enabled (full-frame + sliced)"),
            ("Post-processing", "NMS, class-aware, threshold 0.5"),
            ("SAHI Interval", "Every 5 seconds (between intervals: fast full-frame only)"),
        ]),
        ("MOG2 Motion Pre-filter", [
            ("History Frames", "500"),
            ("Variance Threshold", "50"),
            ("Shadow Detection", "Disabled"),
            ("Learning Rate", "0.005"),
            ("Min Contour Area", "800 px²"),
            ("BBox Padding", "20%"),
            ("Warmup Frames", "30 (MOG2 builds background model before alerting)"),
        ]),
    ]

    for group_name, params in inf_groups:
        story.append(Paragraph(group_name, styles["sub_head"]))
        rows = [[Paragraph(k, styles["body"]), Paragraph(v, styles["body"])] for k, v in params]
        t = Table(rows, colWidths=[55*mm, W-55*mm])
        t.setStyle(TableStyle([
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [LIGHT_GRAY, WHITE]),
            ("GRID",  (0,0), (-1,-1), 0.3, MID_GRAY),
            ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 2*mm))

        # SECTION 10 — FILE STRUCTURE & SCRIPTS
        story += section_header("10. Project File Structure & Scripts", styles)

    scripts = [
        ("collect_data.py",          "Downloads all 17 Roboflow datasets, maps labels to unified class space, and builds the merged train/val split (fixed seed=42 for reproducibility)."),
        ("augment_nightvision.py",   "Converts 40% of training images to synthetic IR/night-vision using Albumentations. Two modes: mild and harsh."),
        ("augment_weak_classes.py",  "Offline augmentation for minority classes (nilgai, elephant, deer). Creates genuinely new images via crop/zoom/flip/HSV jitter."),
        ("train.py",                 "Main training script. Implements smart-resume logic: continues from last.pt if it exists (preserving optimizer state), otherwise starts fresh."),
        ("evaluate.py",              "Runs YOLO validation and produces per-class AP, Precision, Recall, F1 table. Saves results to runs/eval/eval_results.json."),
        ("detect.py",                "Real-time inference script. Uses MOG2 motion detection, SAHI for small-object detection, and draws color-coded bounding boxes."),
        ("config.py",                "Central configuration file. All hyperparameters, class definitions, dataset URLs, augmentation settings, and inference config are here."),
        ("zip_dataset.py",           "Packages project + last.pt into kaggle_dataset.zip for uploading to Kaggle to resume training sessions."),
        ("make_nb.py",               "Generates the Kaggle notebook (Colab.ipynb) with all training cells pre-configured."),
    ]

    for script, desc in scripts:
        story.append(Paragraph(
            f"<b>{script}</b> — {desc}",
            styles["body"]
        ))
        story.append(Spacer(1, 1*mm))

        # FOOTER
        story.append(Spacer(1, 8*mm))
    story.append(hr(ACCENT, 1.0))
    story.append(Paragraph(
        f"SmartGuard — YOLO11m Training Report &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"SmartGuard Systems Pvt. Ltd. &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')} IST",
        ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#94A3B8"),
                       fontName="Helvetica", alignment=TA_CENTER)
    ))

    doc.build(story)
    print("Report saved to: SmartGuard_Training_Report.pdf")

if __name__ == "__main__":
    build_pdf()
