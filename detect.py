import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np─────────────────────────
from config import (
    TARGET_CLASSES,
    CLASS_COLORS,
    MOTION_CONFIG,
    SAHI_CONFIG,
    INFERENCE_CONFIG,
    RUNS_DIR,
)

# MOTION DETECTION MODULE (Path 1)
class MotionDetector:
    """
    OpenCV MOG2-based motion detector that extracts regions of interest
    (ROIs) from video frames where movement is detected.
    """

    def __init__(self, config=None):
        cfg = config or MOTION_CONFIG
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=cfg["history"],
            varThreshold=cfg["var_threshold"],
            detectShadows=cfg["detect_shadows"],
        )
        self.learning_rate = cfg["learning_rate"]
        self.min_area = cfg["min_contour_area"]
        self.bbox_padding = cfg["bbox_padding"]
        self.warmup_frames = cfg["warmup_frames"]
        self.frame_count = 0

        # Morphological kernels
        self.erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg["erode_kernel"], cfg["erode_kernel"])
        )
        self.dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg["dilate_kernel"], cfg["dilate_kernel"])
        )
        self.morph_iterations = cfg["morph_iterations"]

    def detect_motion_rois(self, frame):
        """
        Detect motion regions in a frame and return padded bounding boxes.

        Args:
            frame: BGR image (numpy array)

        Returns:
            List of (x1, y1, x2, y2) bounding boxes in original frame coords
        """
        self.frame_count += 1

        # Apply MOG2 background subtraction
        fg_mask = self.mog2.apply(frame, learningRate=self.learning_rate)

        # Skip during warmup (MOG2 needs time to build background model)
        if self.frame_count < self.warmup_frames:
            return []

        # Morphological filtering: erode (remove noise) → dilate (fill gaps)
        fg_mask = cv2.erode(fg_mask, self.erode_kernel,
                            iterations=self.morph_iterations)
        fg_mask = cv2.dilate(fg_mask, self.dilate_kernel,
                             iterations=self.morph_iterations)

        # Threshold to binary
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = frame.shape[:2]
        rois = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue

            # Get bounding box
            x, y, bw, bh = cv2.boundingRect(contour)

            # Add padding
            pad_x = int(bw * self.bbox_padding)
            pad_y = int(bh * self.bbox_padding)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)

            # Skip very small ROIs
            if (x2 - x1) < 20 or (y2 - y1) < 20:
                continue

            rois.append((x1, y1, x2, y2))

        # Merge overlapping ROIs
        if len(rois) > 1:
            rois = self._merge_overlapping_rois(rois)

        return rois

    def _merge_overlapping_rois(self, rois, iou_threshold=0.3):
        """Merge ROIs that overlap significantly."""
        if not rois:
            return rois

        # Sort by area (largest first)
        rois = sorted(rois, key=lambda r: (r[2]-r[0]) * (r[3]-r[1]), reverse=True)
        merged = []
        used = [False] * len(rois)

        for i, roi_a in enumerate(rois):
            if used[i]:
                continue

            x1, y1, x2, y2 = roi_a

            for j in range(i + 1, len(rois)):
                if used[j]:
                    continue

                # Check overlap
                roi_b = rois[j]
                ix1 = max(x1, roi_b[0])
                iy1 = max(y1, roi_b[1])
                ix2 = min(x2, roi_b[2])
                iy2 = min(y2, roi_b[3])

                if ix1 < ix2 and iy1 < iy2:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area_b = (roi_b[2]-roi_b[0]) * (roi_b[3]-roi_b[1])
                    if area_b > 0 and inter / area_b > iou_threshold:
                        # Merge B into A
                        x1 = min(x1, roi_b[0])
                        y1 = min(y1, roi_b[1])
                        x2 = max(x2, roi_b[2])
                        y2 = max(y2, roi_b[3])
                        used[j] = True

            merged.append((x1, y1, x2, y2))
            used= True

        return merged

# YOLO INFERENCE MODULE
class YOLODetector:
    """Wrapper around the Ultralytics YOLO model for inference."""

    def __init__(self, weights_path, conf=None, iou=None):
        try:
            from ultralytics import YOLO
        except ImportError:
            import os
            os.system(f"{sys.executable} -m pip install ultralytics")
            from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.conf = conf or INFERENCE_CONFIG["conf_threshold"]
        self.iou = iou or INFERENCE_CONFIG["iou_threshold"]
        self.max_det = INFERENCE_CONFIG["max_det"]

    def predict(self, image, imgsz=512):  # 512 matches training resolution
        """
        Run YOLO inference on an image.

        Args:
            image: BGR image (numpy array)
            imgsz: Inference resolution

        Returns:
            List of dicts: [{class_name, confidence, bbox_xyxy}, ...]
        """
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            imgsz=imgsz,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_idx = int(box.cls.item())
                confidence = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Get class name
                if cls_idx < len(TARGET_CLASSES):
                    class_name = TARGET_CLASSES[cls_idx]
                else:
                    class_name = result.names.get(cls_idx, f"class_{cls_idx}")

                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "class_idx": cls_idx,
                })

        return detections

# SAHI DETECTION MODULE (Path 2)
class SAHIDetector:
    """
    Sliced Aided Hyper Inference — processes large frames in overlapping
    tiles to catch small/distant objects that full-frame inference misses.
    """

    def __init__(self, weights_path, conf=None, sahi_config=None):
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self.get_sliced_prediction = get_sliced_prediction
        except ImportError:
            import os
            os.system(f"{sys.executable} -m pip install sahi")
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            self.get_sliced_prediction = get_sliced_prediction

        self.conf = conf or INFERENCE_CONFIG["conf_threshold"]
        self.sahi_cfg = sahi_config or SAHI_CONFIG

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights_path),
            confidence_threshold=self.conf,
            device="cuda:0" if self._has_cuda() else "cpu",
        )

    def _has_cuda(self):
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def predict(self, image):
        """
        Run SAHI sliced inference on an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            List of dicts: [{class_name, confidence, bbox_xyxy}, ...]
        """
        result = self.get_sliced_prediction(
            image=image,
            detection_model=self.detection_model,
            slice_height=self.sahi_cfg["slice_height"],
            slice_width=self.sahi_cfg["slice_width"],
            overlap_height_ratio=self.sahi_cfg["overlap_height_ratio"],
            overlap_width_ratio=self.sahi_cfg["overlap_width_ratio"],
            perform_standard_pred=self.sahi_cfg["perform_standard_pred"],
            postprocess_type=self.sahi_cfg["postprocess_type"],
            postprocess_match_threshold=self.sahi_cfg["postprocess_match_threshold"],
            postprocess_class_agnostic=self.sahi_cfg["postprocess_class_agnostic"],
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox = pred.bbox
            x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy

            class_name = pred.category.name
            confidence = pred.score.value

            # Try to map to our target class index
            class_idx = -1
            if class_name in TARGET_CLASSES:
                class_idx = TARGET_CLASSES.index(class_name)

            detections.append({
                "class_name": class_name,
                "confidence": confidence,
                "bbox_xyxy": (int(x1), int(y1), int(x2), int(y2)),
                "class_idx": class_idx,
            })

        return detections

# DETECTION FUSION
def compute_iou(box_a, box_b):
    """Compute Intersection over Union between two boxes (x1,y1,x2,y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0

class SimpleIoUTracker:
    """
    Lightweight frame-to-frame IoU tracker.
    Tracks detected animals across frames and estimates the count of unique animals.
    """
    def __init__(self, iou_threshold=0.25, max_age=15):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.active_tracks = []  # list of dicts: {"id", "bbox", "class_name", "age", "hits"}
        self.total_unique_counts = defaultdict(int)

    def update(self, detections):
        # Increment age of all active tracks
        for track in self.active_tracks:
            track["age"] += 1

        used_detections = [False] * len(detections)

        # Sort tracks by age (youngest matched first)
        self.active_tracks.sort(key=lambda t: t["age"])

        # 1. Match new detections to existing active tracks
        for track in self.active_tracks:
            best_iou = 0
            best_idx = -1
            for idx, det in enumerate(detections):
                if used_detections[idx]:
                    continue
                if det["class_name"] != track["class_name"]:
                    continue

                iou = compute_iou(track["bbox"], det["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou > self.iou_threshold and best_idx != -1:
                # Match found! Update track state
                track["bbox"] = detections[best_idx]["bbox_xyxy"]
                track["age"] = 0
                track["hits"] += 1
                detections[best_idx]["track_id"] = track["id"]
                used_detections[best_idx] = True

        # 2. Filter out tracks that have been inactive for too long
        self.active_tracks = [t for t in self.active_tracks if t["age"] <= self.max_age]

        # 3. Create new tracks for unmatched detections (new unique animals)
        for idx, det in enumerate(detections):
            if not used_detections[idx]:
                new_track = {
                    "id": self.next_id,
                    "bbox": det["bbox_xyxy"],
                    "class_name": det["class_name"],
                    "age": 0,
                    "hits": 1
                }
                detections[idx]["track_id"] = self.next_id
                self.next_id += 1
                self.total_unique_counts[det["class_name"]] += 1
                self.active_tracks.append(new_track)

def fuse_detections(motion_dets, sahi_dets, iou_threshold=0.5):
    """
    Fuse detections from motion path and SAHI path.
    De-duplicates overlapping detections, keeping the one with higher confidence.

    Args:
        motion_dets: Detections from motion-triggered path
        sahi_dets: Detections from SAHI full-scan path
        iou_threshold: IoU threshold for considering detections as duplicates

    Returns:
        List of fused detections with 'source' field added
    """
    # Tag sources
    for d in motion_dets:
        d["source"] = "motion"
    for d in sahi_dets:
        d["source"] = "sahi"

    if not motion_dets:
        return sahi_dets
    if not sahi_dets:
        return motion_dets

    # Start with all motion detections
    fused = list(motion_dets)
    used_sahi = [False] * len(sahi_dets)

    # For each SAHI detection, check if it overlaps with any motion detection
    for i, sahi_det in enumerate(sahi_dets):
        is_duplicate = False
        for motion_det in motion_dets:
            iou = compute_iou(sahi_det["bbox_xyxy"], motion_det["bbox_xyxy"])
            if iou > iou_threshold:
                # Same object detected by both paths — keep higher confidence
                if sahi_det["confidence"] > motion_det["confidence"]:
                    motion_det.update(sahi_det)
                    motion_det["source"] = "both"
                is_duplicate = True
                break

        if not is_duplicate:
            fused.append(sahi_det)

    return fused

# VISUALIZATION
def draw_detections(frame, detections, show_source=False):
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame: BGR image (numpy array)
        detections: List of detection dicts
        show_source: If True, show detection source (motion/sahi/both)

    Returns:
        Annotated frame
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        source = det.get("source", "")

        # Get class color
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Build label text
        label = f"{class_name} {confidence:.2f}"
        if "track_id" in det:
            label = f"#{det['track_id']} {label}"
        if show_source and source:
            label += f" [{source}]"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Draw label text
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return annotated

# MAIN DETECTION PIPELINE
class AnimalDetector:
    """
    The main dual-path detection pipeline that orchestrates
    motion detection, SAHI tiling, and YOLO inference.
    """

    def __init__(self, weights_path, mode="dual",
                 sahi_interval=None, conf=None):
        """
        Args:
            weights_path: Path to trained YOLO weights (.pt file)
            mode: "dual" (motion + SAHI), "motion" (motion only), "sahi" (SAHI only)
            sahi_interval: Seconds between SAHI full-frame scans (for video)
            conf: Confidence threshold override
        """
        self.mode = mode
        self.sahi_interval = sahi_interval or INFERENCE_CONFIG["sahi_interval_seconds"]

        # Initialize YOLO detector (always needed)
        self.yolo = YOLODetector(weights_path, conf=conf)

        # Initialize motion detector (for motion & dual modes)
        if mode in ("dual", "motion"):
            self.motion = MotionDetector()
        else:
            self.motion = None

        # Initialize SAHI detector (for sahi & dual modes)
        if mode in ("dual", "sahi"):
            self.sahi = SAHIDetector(weights_path, conf=conf)
        else:
            self.sahi = None

        self.last_sahi_time = 0
        self.tracker = SimpleIoUTracker(iou_threshold=0.25, max_age=15)

    def detect_image(self, image, show_source=False):
        """
        Run detection on a single image.

        For images, we always use SAHI (no motion context available).
        If mode is "motion", we fall back to direct YOLO inference.

        Args:
            image: BGR image (numpy array)
            show_source: Add source labels to detections

        Returns:
            (annotated_frame, detections_list)
        """
        if self.mode == "motion":
            # No motion context for single image — direct YOLO
            detections = self.yolo.predict(image, imgsz=512)
            for d in detections:
                d["source"] = "direct"
        elif self.mode == "sahi":
            detections = self.sahi.predict(image)
            for d in detections:
                d["source"] = "sahi"
        else:  # dual
            # For single image, use SAHI (more thorough than motion)
            sahi_dets = self.sahi.predict(image)
            direct_dets = self.yolo.predict(image, imgsz=512)
            for d in direct_dets:
                d["source"] = "direct"
            for d in sahi_dets:
                d["source"] = "sahi"
            detections = fuse_detections(direct_dets, sahi_dets)

        annotated = draw_detections(image, detections, show_source=show_source)
        return annotated, detections

    def detect_video_frame(self, frame, current_time=None):
        """
        Run detection on a single video frame.

        Motion detection runs on every frame.
        SAHI full-scan runs periodically based on sahi_interval.

        Args:
            frame: BGR video frame
            current_time: Current timestamp in seconds (for SAHI scheduling)

        Returns:
            (annotated_frame, detections_list)
        """
        if current_time is None:
            current_time = time.time()

        motion_dets = []
        sahi_dets = []

        # ── Path 1: Motion-triggered detection ──
        if self.motion is not None:
            rois = self.motion.detect_motion_rois(frame)

            for roi in rois:
                x1, y1, x2, y2 = roi
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                # Run YOLO on cropped ROI
                crop_dets = self.yolo.predict(crop)

                # Remap coordinates back to full frame
                for det in crop_dets:
                    bx1, by1, bx2, by2 = det["bbox_xyxy"]
                    det["bbox_xyxy"] = (
                        bx1 + x1, by1 + y1,
                        bx2 + x1, by2 + y1,
                    )
                    det["source"] = "motion"

                motion_dets.extend(crop_dets)

        # ── Path 2: Periodic SAHI full-frame scan ──
        if self.sahi is not None:
            time_since_last = current_time - self.last_sahi_time
            should_scan = (
                time_since_last >= self.sahi_interval
                or (not motion_dets and self.motion is not None)  # No motion found
            )

            if should_scan:
                sahi_dets = self.sahi.predict(frame)
                for d in sahi_dets:
                    d["source"] = "sahi"
                self.last_sahi_time = current_time

        # ── Fuse detections ──
        if self.mode == "motion":
            detections = motion_dets
        elif self.mode == "sahi":
            detections = sahi_dets
        else:
            detections = fuse_detections(motion_dets, sahi_dets)self.tracker.update(detections)

        annotated = draw_detections(frame, detections, show_source=True)
        return annotated, detections

def process_source(detector, source, save_video=False, save_dir=None):
    """
    Process an image, video, directory, or webcam stream.

    Args:
        detector: AnimalDetector instance
        source: Path to image/video/directory, or camera index (0, 1, ...)
        save_video: Save annotated video output
        save_dir: Directory to save outputs
    """
    source = str(source)

    # Create output directory
    if save_dir is None:
        save_dir = RUNS_DIR / "detect"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine source type ──
    if source.isdigit():
        # Webcam
        _process_video(detector, int(source), save_video, save_dir, is_webcam=True)
    elif Path(source).is_dir():
        # Directory of images
        _process_directory(detector, Path(source), save_dir)
    elif Path(source).suffix.lower() in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
        # Video file
        _process_video(detector, source, save_video, save_dir)
    else:
        # Single image
        _process_image(detector, source, save_dir)

def _process_image(detector, image_path, save_dir):
    """Process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Processing: {image_path}")
    annotated, detections = detector.detect_image(image, show_source=True)

    # Save result
    out_path = save_dir / f"det_{Path(image_path).name}"
    cv2.imwrite(str(out_path), annotated)

    # Print detections
    if detections:
        print(f"  Found {len(detections)} detection(s):")
        for d in detections:
            print(f"    - {d['class_name']} ({d['confidence']:.2f}) "
                  f"@ {d['bbox_xyxy']} [{d.get('source', '')}]")
    else:
        print(f"  No detections.")

    print(f"  Saved: {out_path}")

def _process_directory(detector, dir_path, save_dir):
    """Process all images in a directory."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([
        f for f in dir_path.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not images:
        print(f"ERROR: No images found in {dir_path}")
        return

    print(f"Processing {len(images)} images from {dir_path}")
    total_dets = defaultdict(int)

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        annotated, detections = detector.detect_image(image, show_source=True)

        # Save
        out_path = save_dir / f"det_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        for d in detections:
            total_dets[d["class_name"]] += 1

    print(f"\n  Detection summary across {len(images)} images:")
    for cls, count in sorted(total_dets.items()):
        print(f"    {cls}: {count}")
    print(f"  Results saved to: {save_dir}")

def _process_video(detector, source, save_video, save_dir, is_webcam=False):
    """Process a video file or webcam stream."""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0

    print(f"Video: {w}x{h} @ {fps:.1f} FPS")
    if total_frames > 0:
        print(f"Total frames: {total_frames}")

    # Video writer
    writer = None
    if save_video:
        out_path = save_dir / f"det_{Path(str(source)).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Saving to: {out_path}")

    frame_idx = 0
    start_time = time.time()
    total_dets = defaultdict(int)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = frame_idx / fps  # Simulated time from frame index

            # Run detection
            annotated, detections = detector.detect_video_frame(
                frame, current_time=current_time
            )

            # Track stats
            for d in detections:
                total_dets[d["class_name"]] += 1

            # Add FPS counter
            elapsed = time.time() - start_time
            current_fps = frame_idx / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated,
                f"FPS: {current_fps:.1f} | Detections: {len(detections)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            # Save frame
            if writer is not None:
                writer.write(annotated)

            # Display (skip on headless servers / Colab)
            try:
                cv2.imshow("Animal Guard Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break
            except cv2.error:
                # Headless environment — just process frames
                pass

            # Progress update every 100 frames
            if frame_idx % 100 == 0:
                progress = f" ({frame_idx}/{total_frames})" if total_frames > 0 else ""
                print(f"  Frame {frame_idx}{progress} | "
                      f"FPS: {current_fps:.1f} | "
                      f"Detections this frame: {len(detections)}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    # Final summary
    elapsed = time.time() - start_time
    avg_fps = frame_idx / elapsed if elapsed > 0 else 0
    print(f"\n{'─' * 50}")
    print(f"  Processed {frame_idx} frames in {elapsed:.1f}s ({avg_fps:.1f} FPS)")
    print(f"  Detection summary (total frame-level occurrences):")
    for cls, count in sorted(total_dets.items()):
        print(f"    {cls}: {count} occurrences across all frames")

    # Unique animal count from tracker
    print(f"\n  Estimated unique animals seen in video:")
    unique_counts = detector.tracker.total_unique_counts
    if unique_counts:
        for cls, count in sorted(unique_counts.items()):
            print(f"    {cls}: {count} unique animal(s)")
    else:
        print("    None")
    print(f"{'─' * 50}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Animal Guard — Dual-path animal detection pipeline"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Image, video, directory, or camera index (0, 1, ...)"
    )
    parser.add_argument(
        "--weights", type=str, required=True,
        help="Path to trained YOLO weights (.pt)"
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help=f"Confidence threshold (default: {INFERENCE_CONFIG['conf_threshold']})"
    )
    parser.add_argument(
        "--sahi-only", action="store_true",
        help="Use only SAHI tiled inference (skip motion detection)"
    )
    parser.add_argument(
        "--motion-only", action="store_true",
        help="Use only motion-triggered inference (skip SAHI)"
    )
    parser.add_argument(
        "--sahi-interval", type=float, default=None,
        help=f"Seconds between SAHI scans (default: {INFERENCE_CONFIG['sahi_interval_seconds']})"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save annotated video output"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output directory for results"
    )
    args = parser.parse_args()

    # Determine mode
    if args.sahi_only and args.motion_only:
        print("ERROR: Cannot use both --sahi-only and --motion-only")
        sys.exit(1)
    elif args.sahi_only:
        mode = "sahi"
    elif args.motion_only:
        mode = "motion"
    else:
        mode = "dual"

    print(f"\n{'═' * 60}")
    print(f"  ANIMAL GUARD DETECTION")
    print(f"  Mode: {mode.upper()}")
    print(f"  Weights: {args.weights}")
    print(f"  Source: {args.source}")
    print(f"{'═' * 60}\n")

    # Initialize detector
    detector = AnimalDetector(
        weights_path=args.weights,
        mode=mode,
        sahi_interval=args.sahi_interval,
        conf=args.conf,
    )

    # Process source
    process_source(
        detector,
        args.source,
        save_video=args.save_video,
        save_dir=args.save_dir,
    )

if __name__ == "__main__":
    main()
