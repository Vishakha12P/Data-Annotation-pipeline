"""
Data annotation pipeline using YOLO (pretrained, for "person") +
YOLO-World (open-vocabulary, for custom PPE classes).

No custom C++/CUDA extensions to compile — GPU acceleration works
automatically through plain PyTorch/ultralytics if CUDA is available.

Usage:
    python annotate.py
    python annotate.py --config my_config.yaml
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import yaml
import torch
from tqdm import tqdm
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Config / IO helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_video_list(path: str) -> list:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video list file not found: {path}")
    videos = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not os.path.isfile(line):
                print(f"  Warning: video not found, skipping: {line}")
                continue
            videos.append(line)
    return videos


def ensure_dirs(out_cfg: dict):
    os.makedirs(out_cfg["frames_dir"], exist_ok=True)
    os.makedirs(out_cfg["primary_labels_dir"], exist_ok=True)
    os.makedirs(out_cfg["secondary_images_dir"], exist_ok=True)
    os.makedirs(out_cfg["secondary_labels_dir"], exist_ok=True)
    os.makedirs(out_cfg["secondary_metadata_dir"], exist_ok=True)
    if out_cfg.get("save_visualization", True):
        os.makedirs(out_cfg["primary_viz_dir"], exist_ok=True)
        os.makedirs(out_cfg["secondary_viz_dir"], exist_ok=True)


def save_yolo_labels(path: str, detections: list):
    """detections: list of (class_id, x_center, y_center, w, h), all normalized."""
    with open(path, "w") as f:
        for cid, x, y, w, h in detections:
            f.write(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def create_dataset_yaml(path: str, class_names: list, data_root: str = "."):
    cfg = {
        "path": data_root,
        "train": "images",
        "val": "images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def draw_boxes(image_bgr, boxes_xyxy, labels, scores, colors=None):
    img = image_bgr.copy()
    for box, label, score in zip(boxes_xyxy, labels, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


# ---------------------------------------------------------------------------
# Frame extraction (plain OpenCV, no ffmpeg dependency)
# ---------------------------------------------------------------------------

def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    cap.release()
    return info


def frame_generator(video_path: str, skip_frames: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    step = skip_frames + 1
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                yield idx, frame
            idx += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Core per-frame processing
# ---------------------------------------------------------------------------

def crop_with_padding(image, box_xyxy, padding: float):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1
    pad_w, pad_h = bw * padding, bh * padding
    x1p = max(0, int(x1 - pad_w))
    y1p = max(0, int(y1 - pad_h))
    x2p = min(w, int(x2 + pad_w))
    y2p = min(h, int(y2 + pad_h))
    crop = image[y1p:y2p, x1p:x2p].copy()
    crop_info = {
        "offset_x": x1p, "offset_y": y1p,
        "crop_width": x2p - x1p, "crop_height": y2p - y1p,
        "original_width": w, "original_height": h,
    }
    return crop, crop_info


def process_frame(
    frame_bgr,
    frame_name: str,
    person_model: YOLO,
    ppe_model: YOLO,
    config: dict,
    stats: dict,
    device: str,
) -> bool:
    """
    Run person detection on the full frame, then PPE detection on each
    person crop. Saves YOLO labels + optional visualizations.

    Returns True if at least one person was detected.
    """
    out_cfg = config["output"]
    primary_cfg = config["primary"]
    secondary_cfg = config["secondary"]
    save_viz = out_cfg.get("save_visualization", True)

    img_h, img_w = frame_bgr.shape[:2]

    # ===== PRIMARY: person detection =====
    results = person_model.predict(
        frame_bgr,
        conf=primary_cfg["conf_threshold"],
        classes=[0],           # class 0 = "person" in COCO
        device=device,
        verbose=False,
    )[0]

    person_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
    person_scores = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

    primary_yolo_dets = []
    for box, score in zip(person_boxes, person_scores):
        x1, y1, x2, y2 = box
        x_c = ((x1 + x2) / 2.0) / img_w
        y_c = ((y1 + y2) / 2.0) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        primary_yolo_dets.append((0, x_c, y_c, w, h))
        stats["primary"]["person"] += 1

    has_person = len(primary_yolo_dets) > 0
    if not has_person:
        return False

    label_path = os.path.join(out_cfg["primary_labels_dir"], f"{frame_name}.txt")
    save_yolo_labels(label_path, primary_yolo_dets)

    if save_viz:
        viz = draw_boxes(
            frame_bgr, person_boxes,
            ["person"] * len(person_boxes), person_scores
        )
        cv2.imwrite(os.path.join(out_cfg["primary_viz_dir"], f"{frame_name}.jpg"), viz)

    # ===== SECONDARY: PPE detection per person crop =====
    ppe_classes = secondary_cfg["classes"]
    crop_padding = secondary_cfg.get("crop_padding", 0.15)

    for crop_idx, box in enumerate(person_boxes):
        crop, crop_info = crop_with_padding(frame_bgr, box, crop_padding)
        if crop.size == 0:
            continue

        ppe_results = ppe_model.predict(
            crop,
            conf=secondary_cfg["conf_threshold"],
            device=device,
            verbose=False,
        )[0]

        ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy() if ppe_results.boxes is not None else []
        ppe_scores = ppe_results.boxes.conf.cpu().numpy() if ppe_results.boxes is not None else []
        ppe_cls_ids = ppe_results.boxes.cls.cpu().numpy().astype(int) if ppe_results.boxes is not None else []

        if len(ppe_boxes) == 0:
            continue

        crop_h, crop_w = crop.shape[:2]
        yolo_dets = []
        valid_boxes, valid_labels, valid_scores = [], [], []

        for box_c, score, cid in zip(ppe_boxes, ppe_scores, ppe_cls_ids):
            x1, y1, x2, y2 = box_c
            x_c = ((x1 + x2) / 2.0) / crop_w
            y_c = ((y1 + y2) / 2.0) / crop_h
            w = (x2 - x1) / crop_w
            h = (y2 - y1) / crop_h
            yolo_dets.append((cid, x_c, y_c, w, h))
            valid_boxes.append(box_c)
            valid_labels.append(ppe_classes[cid] if cid < len(ppe_classes) else str(cid))
            valid_scores.append(score)
            stats["secondary"][ppe_classes[cid] if cid < len(ppe_classes) else str(cid)] += 1

        crop_name = f"{frame_name}_person{crop_idx:02d}"
        crop_path = os.path.join(out_cfg["secondary_images_dir"], f"{crop_name}.jpg")
        cv2.imwrite(crop_path, crop)

        save_yolo_labels(
            os.path.join(out_cfg["secondary_labels_dir"], f"{crop_name}.txt"),
            yolo_dets
        )

        with open(os.path.join(out_cfg["secondary_metadata_dir"], f"{crop_name}.json"), "w") as f:
            json.dump({"parent_frame": frame_name, "crop_index": crop_idx, **crop_info}, f, indent=2)

        if save_viz:
            viz = draw_boxes(crop, valid_boxes, valid_labels, valid_scores)
            cv2.imwrite(os.path.join(out_cfg["secondary_viz_dir"], f"{crop_name}.jpg"), viz)

    return True


# ---------------------------------------------------------------------------
# Video mode driver (with resume support)
# ---------------------------------------------------------------------------

def run_video_mode(config: dict, person_model: YOLO, ppe_model: YOLO, device: str):
    video_cfg = config["video"]
    out_cfg = config["output"]
    skip_frames = video_cfg.get("skip_frames", 0)

    video_paths = parse_video_list(video_cfg["video_list"])
    print(f"Found {len(video_paths)} video(s) to process")

    stats = {"primary": defaultdict(int), "secondary": defaultdict(int)}
    total_processed = 0
    total_kept = 0
    total_discarded = 0

    for video_path in video_paths:
        video_name = Path(video_path).stem
        try:
            info = get_video_info(video_path)
        except Exception as e:
            print(f"  Skipping (cannot open): {video_path} — {e}")
            continue

        est_frames = info["total_frames"] // (skip_frames + 1)
        print(f"\n> {Path(video_path).name}  "
              f"({info['width']}x{info['height']}, {info['fps']:.1f}fps, "
              f"{info['duration_sec']:.1f}s, ~{est_frames} frames to check)")

        # Resume notice
        existing = len([
            f for f in os.listdir(out_cfg["primary_labels_dir"])
            if f.startswith(video_name) and f.endswith(".txt")
        ]) if os.path.isdir(out_cfg["primary_labels_dir"]) else 0
        if existing:
            print(f"  Resuming: {existing} frame(s) already labeled, will be skipped.")

        for frame_idx, frame_bgr in tqdm(
            frame_generator(video_path, skip_frames),
            total=est_frames, desc=f"  {video_name}"
        ):
            total_processed += 1
            frame_name = f"{video_name}_frame_{frame_idx:06d}"

            label_path = os.path.join(out_cfg["primary_labels_dir"], f"{frame_name}.txt")
            if os.path.isfile(label_path):
                total_kept += 1
                continue

            try:
                kept = process_frame(
                    frame_bgr, frame_name, person_model, ppe_model,
                    config, stats, device
                )
            except Exception as e:
                print(f"\n  Error on frame {frame_idx}: {e}")
                continue

            if kept:
                total_kept += 1
            else:
                total_discarded += 1

    # Dataset YAMLs
    if out_cfg.get("create_dataset_yaml", True):
        create_dataset_yaml("dataset_primary.yaml", ["person"], data_root="data/primary")
        create_dataset_yaml(
            "dataset_secondary.yaml",
            config["secondary"]["classes"],
            data_root="data/secondary"
        )
        print("\nCreated dataset_primary.yaml and dataset_secondary.yaml")

    print("\n" + "=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"Frames checked   : {total_processed}")
    print(f"Frames kept      : {total_kept}")
    print(f"Frames discarded : {total_discarded}")
    print(f"\nPrimary detections:")
    for name, count in stats["primary"].items():
        print(f"  {name}: {count}")
    print(f"\nSecondary (PPE) detections:")
    for name, count in stats["secondary"].items():
        print(f"  {name}: {count}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YOLO + YOLO-World annotation pipeline")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs(config["output"])

    device = config["model"].get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — falling back to CPU")
        device = "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading person detection model ({config['model']['person_model']})...")
    person_model = YOLO(config["model"]["person_model"])

    print(f"Loading PPE detection model ({config['model']['ppe_model']})...")
    ppe_model = YOLO(config["model"]["ppe_model"])
    ppe_model.set_classes(config["secondary"]["classes"])

    print("Models loaded.\n")

    run_video_mode(config, person_model, ppe_model, device)


if __name__ == "__main__":
    main()
