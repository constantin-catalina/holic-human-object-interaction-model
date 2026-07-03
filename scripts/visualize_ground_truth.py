"""
visualize_ground_truth.py
Genereaza video-uri annotate cu ground truth pentru videoclipurile MPHOI-72.

Usage:
    python scripts/visualize_ground_truth.py \
        --video-id Subject12-task_1_cheering-take_1 \
        --data-root data/mphoi72/ \
        --input mphoi72-videos/Subject12-task_1_cheering-take_1.mp4 \
        --output outputs/gt_viz/Subject12-task_1_cheering-take_1_gt.mp4
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import json
import argparse
import numpy as np
import cv2

from utils.video_annotation import COLORS, get_label_names, draw_predictions_on_frame, draw_timeline, save_visualization_video
from data.preprocess import VideoReader


def load_ground_truth_json(gt_path, num_frames):
    """
    Incarca ground truth din formatul MPHOI-72 JSON:
    { "Human1": [start, label, end, label, ...], ... }
    Returneaza np.array (S, M) cu label ID per frame per entitate.
    """
    with open(gt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    humans = sorted([k for k in data.keys() if k.startswith('Human')])
    M = len(humans)
    gt = np.full((num_frames, M), -1, dtype=np.int64)

    for m, human in enumerate(humans):
        seq = data[human]
        # Format: [start, label, end, label, ...]
        for i in range(0, len(seq) - 1, 2):
            start = seq[i]
            label = seq[i + 1]
            end = seq[i + 2] if i + 2 < len(seq) else num_frames
            gt[start:end, m] = label

    return gt


def main():
    parser = argparse.ArgumentParser(description="Vizualizare Ground Truth MPHOI-72")
    parser.add_argument("--video-id", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/mphoi72/")
    parser.add_argument("--input", type=str, required=True, help="Path catre fisierul video")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    # Ground truth path
    parts = args.video_id.split('-')
    subject = parts[0]
    task = '-'.join(parts[1:-1])  # e.g., task_1_cheering
    take = parts[-1]
    gt_path = os.path.join(args.data_root, "Human_subactivity_ground_truth", subject, task, f"{take}.json")

    if not os.path.exists(gt_path):
        print(f"[EROARE] Ground truth nu exista: {gt_path}")
        return

    # Incarca video
    reader = VideoReader(args.input, max_frames=args.max_frames)
    frames = reader.read_frames()
    S = len(frames)
    H, W = frames[0].shape[:2]

    # Incarca ground truth
    gt_labels = load_ground_truth_json(gt_path, S)
    M = gt_labels.shape[1]

    # Incarca bounding boxes din features (daca exista)
    feat_dir = os.path.join(args.data_root, "features")
    bbox_path = os.path.join(feat_dir, f"{args.video_id}_bbox.npy")
    bboxes = None
    if os.path.exists(bbox_path):
        bboxes = np.load(bbox_path).astype(np.float32)
        if len(bboxes) > S:
            bboxes = bboxes[:S]
        elif len(bboxes) < S:
            missing = S - len(bboxes)
            bboxes = np.concatenate([bboxes, np.tile(bboxes[-1:], (missing, 1, 1))], axis=0)
    else:
        print(f"[WARN] Bbox-uri lipsa: {bbox_path}")

    # Entity types
    etypes_path = os.path.join(feat_dir, f"{args.video_id}_entity_types.npy")
    entity_types = None
    if os.path.exists(etypes_path):
        entity_types = np.load(etypes_path).astype(np.int64)
    else:
        entity_types = np.array([0] * M, dtype=np.int64)

    # Label names
    label_names = get_label_names("mphoi72")

    # Output path
    if args.output is None:
        out_dir = os.path.join("outputs", "gt_viz")
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f"{args.video_id}_gt.mp4")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Genereaza video cu timeline
    print(f"Generare video GT: {args.output}")
    print(f"  Frames: {S}, Entities: {M}, Resolution: {W}x{H}")

    # Draw timeline
    timeline = draw_timeline(gt_labels, label_names, width=W)
    tl_h = timeline.shape[0]
    out_h = H + tl_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (W, out_h))

    for s in range(S):
        vis = draw_predictions_on_frame(
            frames[s],
            bboxes[s] if bboxes is not None else np.zeros((M, 4)),
            gt_labels[s],
            label_names,
            s, S,
            source="ground truth",
            entity_types=entity_types,
        )
        tl = timeline.copy()
        cursor_x = 80 + int(s / max(S - 1, 1) * (W - 80))
        cv2.line(tl, (cursor_x, 0), (cursor_x, tl_h), (0, 0, 0), 2)
        tl_resized = cv2.resize(tl, (W, tl_h))
        writer.write(np.vstack([vis, tl_resized]))

    writer.release()
    print(f"[OK] Salvat: {args.output}")


if __name__ == "__main__":
    main()
