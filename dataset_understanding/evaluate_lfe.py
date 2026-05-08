"""
evaluate_lfe.py
Evaluates the trained LFE segmentation model on the FROG test set.

Produces:
    - detections.npz        : inference results in FROG benchmark format
    - evaluation_report.txt : human-readable summary of results
    - detection_samples.png : visualization of sample detections

Usage:
    python evaluate_lfe.py \
        --test_file "dataset/Testing set (16:41)/frog_16-41_test.h5" \
        --weights checkpoints/best_lfe.pt

Then run the official FROG benchmark:
    python calc_pr_curve.py   (from the FROG repo)
    python benchmark.py
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from torch.utils.data import DataLoader
import h5py

from frog_dataset import FrogDataset, LASER_NUM_POINTS, MAX_RANGE
from lfe_model    import LFESegmentation


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
BATCH_SIZE       = 64
NUM_WORKERS      = 4
DETECTION_THRESH = 0.5      # sigmoid threshold for person points
PERSON_RADIUS    = 0.5      # meters — standard FROG circle radius
NMS_DISTANCE     = 0.8      # meters — merge detections closer than this
PEAK_HEIGHT      = 0.3      # minimum peak height in segmentation signal
PEAK_DISTANCE    = 5        # minimum distance between peaks (in points)
NUM_VIZ_SCANS    = 6        # number of scans to visualize


# ─────────────────────────────────────────────
# Post-processing: segmentation → detections
# ─────────────────────────────────────────────

def segmentation_to_detections(probs, raw_scan):
    """
    Convert per-point probabilities into a list of (score, x, y) detections.

    Strategy (LFE-Peaks from the paper):
        1. Find peaks in the probability signal
        2. Each peak corresponds to a candidate person
        3. Compute centroid of nearby high-probability points
        4. Apply NMS to merge nearby detections

    Args:
        probs    : (720,) numpy array of per-point probabilities
        raw_scan : (720,) numpy array of range values in meters

    Returns:
        List of (score, x, y) tuples
    """
    angles = np.linspace(-np.pi / 2, np.pi / 2, LASER_NUM_POINTS)

    # Find peaks in probability signal
    peaks, properties = find_peaks(
        probs,
        height=PEAK_HEIGHT,
        distance=PEAK_DISTANCE
    )

    if len(peaks) == 0:
        return []

    detections = []
    for peak_idx in peaks:
        score = probs[peak_idx]

        # Gather nearby high-probability points as the "cluster"
        window = 10  # points either side
        start  = max(0, peak_idx - window)
        end    = min(LASER_NUM_POINTS, peak_idx + window)

        cluster_mask = probs[start:end] > DETECTION_THRESH
        if cluster_mask.sum() == 0:
            # Fall back to just the peak point
            cluster_mask = np.zeros(end - start, dtype=bool)
            cluster_mask[peak_idx - start] = True

        cluster_ranges = raw_scan[start:end][cluster_mask]
        cluster_angles = angles[start:end][cluster_mask]

        # Filter out invalid ranges
        valid = (cluster_ranges > 0) & (cluster_ranges < MAX_RANGE)
        if valid.sum() == 0:
            continue

        cluster_ranges = cluster_ranges[valid]
        cluster_angles = cluster_angles[valid]

        # Cartesian centroid
        xs = cluster_ranges * np.cos(cluster_angles)
        ys = cluster_ranges * np.sin(cluster_angles)
        cx = xs.mean()
        cy = ys.mean()

        detections.append((float(score), float(cx), float(cy)))

    # Non-maximum suppression — merge detections that are too close
    detections = nms(detections, NMS_DISTANCE)
    return detections


def nms(detections, min_distance):
    """
    Simple distance-based NMS.
    Keeps the highest-scoring detection when two are within min_distance.
    """
    if len(detections) == 0:
        return []

    # Sort by score descending
    detections = sorted(detections, key=lambda d: d[0], reverse=True)
    kept = []

    for det in detections:
        score, x, y = det
        too_close = False
        for _, kx, ky in kept:
            dist = np.sqrt((x - kx) ** 2 + (y - ky) ** 2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            kept.append(det)

    return kept


# ─────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────

def match_detections_to_gt(detections, gt_circles, assoc_dist=0.5):
    """
    Match detections to ground truth annotations by nearest distance.
    Returns (tp, fp, fn) counts.
    """
    if len(gt_circles) == 0:
        return 0, len(detections), 0
    if len(detections) == 0:
        return 0, 0, len(gt_circles)

    matched_gt = set()
    tp, fp = 0, 0

    for score, dx, dy in sorted(detections, key=lambda d: d[0], reverse=True):
        best_dist = float('inf')
        best_gt   = -1

        for gi, ann in enumerate(gt_circles):
            if gi in matched_gt:
                continue
            dist = np.sqrt((dx - ann[0]) ** 2 + (dy - ann[1]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_gt   = gi

        if best_dist <= assoc_dist and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_circles) - len(matched_gt)
    return tp, fp, fn


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def visualize_detections(scans, probs_list, detections_list,
                         gt_circles_list, save_path):
    """Plot a grid of scans showing detections vs ground truth."""
    n    = min(NUM_VIZ_SCANS, len(scans))
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()

    angles = np.linspace(-np.pi / 2, np.pi / 2, LASER_NUM_POINTS)

    for i in range(n):
        ax   = axes[i]
        scan = scans[i]

        # Laser points
        valid = (scan > 0) & (scan < MAX_RANGE)
        xs    = scan[valid] * np.cos(angles[valid])
        ys    = scan[valid] * np.sin(angles[valid])
        ax.scatter(xs, ys, s=2, c='steelblue', alpha=0.5, label='LiDAR')

        # Ground truth (green)
        for ann in gt_circles_list[i]:
            c = plt.Circle((ann[0], ann[1]), PERSON_RADIUS,
                            color='limegreen', fill=False, linewidth=2)
            ax.add_patch(c)

        # Detections (red)
        for score, dx, dy in detections_list[i]:
            c = plt.Circle((dx, dy), PERSON_RADIUS,
                            color='red', fill=False, linewidth=2, linestyle='--')
            ax.add_patch(c)
            ax.text(dx, dy + PERSON_RADIUS + 0.1, f'{score:.2f}',
                    color='red', fontsize=6, ha='center')

        ax.set_xlim(-MAX_RANGE, MAX_RANGE)
        ax.set_ylim(-MAX_RANGE, MAX_RANGE)
        ax.set_aspect('equal')
        ax.set_title(f"Scan {i} | GT: {len(gt_circles_list[i])}  "
                     f"Det: {len(detections_list[i])}", fontsize=9)
        ax.plot(0, 0, 'k^', markersize=8)

    # Legend
    gt_patch  = mpatches.Patch(color='limegreen', label='Ground truth')
    det_patch = mpatches.Patch(color='red',       label='Detection')
    axes[0].legend(handles=[gt_patch, det_patch], fontsize=7, loc='upper right')

    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("LFE Detections vs Ground Truth\n"
                 "(Green = GT, Red dashed = Model detection)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str,
                        default='dataset/Testing set (16:41)/frog_16-41_test.h5',
                        help='Path to FROG test HDF5 file')
    parser.add_argument('--weights', type=str,
                        default='checkpoints/best_lfe.pt',
                        help='Path to trained model weights')
    parser.add_argument('--out_dir', type=str,
                        default='results',
                        help='Directory to save results')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Device ───────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ── Load model ───────────────────────────
    print(f"Loading weights from: {args.weights}")
    model = LFESegmentation().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("Model loaded.")

    # ── Load test dataset ────────────────────
    print(f"\nLoading test set: {args.test_file}")
    test_ds = FrogDataset(args.test_file, split='test', augment=False)
    loader  = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS,
                         pin_memory=True)
    print(f"Test scans: {len(test_ds):,}")

    # ── Inference ────────────────────────────
    print("\nRunning inference...")
    all_probs = []

    with torch.no_grad():
        for scans_batch, _ in loader:
            scans_batch = scans_batch.to(device)
            logits      = model(scans_batch)          # (B, 720)
            probs       = torch.sigmoid(logits)       # (B, 720)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)     # (N, 720)
    print(f"Inference complete. Output shape: {all_probs.shape}")

    # ── Post-processing: seg → detections ────
    print("\nConverting segmentation to detections...")
    all_detections = []
    raw_scans      = test_ds.scans * MAX_RANGE   # denormalise

    for i in range(len(test_ds)):
        dets = segmentation_to_detections(all_probs[i], raw_scans[i])
        all_detections.append(dets)

    # ── Build FROG benchmark format ──────────
    # circles: (M, 3) — score, x, y
    # circle_idx, circle_num: (N,)
    circles_list  = []
    circle_idx    = np.zeros(len(test_ds), dtype=np.uint32)
    circle_num    = np.zeros(len(test_ds), dtype=np.uint32)
    running_idx   = 0

    for i, dets in enumerate(all_detections):
        circle_idx[i] = running_idx
        circle_num[i] = len(dets)
        for score, x, y in dets:
            circles_list.append([score, x, y])
        running_idx += len(dets)

    circles_arr = np.array(circles_list, dtype=np.float32) \
                  if circles_list else np.zeros((0, 3), dtype=np.float32)

    npz_path = os.path.join(args.out_dir, 'detections.npz')
    np.savez(npz_path,
             circles    = circles_arr,
             circle_idx = circle_idx,
             circle_num = circle_num)
    print(f"  Saved benchmark file: {npz_path}")
    print(f"  Total detections: {len(circles_list):,}")

    # ── Quick evaluation ─────────────────────
    print("\nCalculating quick metrics (assoc_dist=0.5m)...")
    total_tp, total_fp, total_fn = 0, 0, 0

    with h5py.File(args.test_file, 'r') as f:
        gt_circle_idx = f['circle_idx'][:]
        gt_circle_num = f['circle_num'][:]
        gt_circles    = f['circles'][:]

    for i, orig_idx in enumerate(test_ds.original_indices):
        idx = gt_circle_idx[orig_idx]
        num = gt_circle_num[orig_idx]
        gt  = gt_circles[idx: idx + num]
        tp, fp, fn = match_detections_to_gt(all_detections[i], gt, assoc_dist=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall    = total_tp / (total_tp + total_fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)

    avg_dets_per_scan = len(circles_list) / len(test_ds)

    report = f"""
====================================================
LFE Evaluation Report — FROG Test Set
====================================================
Test scans evaluated  : {len(test_ds):,}
Total detections made : {len(circles_list):,}
Avg detections/scan   : {avg_dets_per_scan:.2f}

--- Quick Metrics (assoc_dist = 0.5m) ---
True Positives        : {total_tp:,}
False Positives       : {total_fp:,}
False Negatives       : {total_fn:,}
Precision             : {precision:.4f}
Recall                : {recall:.4f}
F1 Score              : {f1:.4f}

--- Files ---
Benchmark detections  : {npz_path}
Visualizations        : {args.out_dir}/detection_samples.png

--- Next Steps ---
Run the official FROG benchmark:
    python calc_pr_curve.py
    python benchmark.py
====================================================
"""
    print(report)

    report_path = os.path.join(args.out_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Saved report: {report_path}")

    # ── Visualize sample detections ──────────
    print("\nGenerating visualizations...")

    # Pick scans spread across the test set for variety
    viz_indices = np.linspace(0, len(test_ds) - 1,
                              NUM_VIZ_SCANS, dtype=int)

    viz_scans      = raw_scans[viz_indices]
    viz_probs      = all_probs[viz_indices]
    viz_detections = [all_detections[i] for i in viz_indices]
    viz_gt = []
    for i in viz_indices:
        idx = test_ds.circle_idx[i]
        num = test_ds.circle_num[i]
        viz_gt.append(test_ds.circles[idx: idx + num])

    viz_path = os.path.join(args.out_dir, 'detection_samples.png')
    visualize_detections(viz_scans, viz_probs, viz_detections,
                         viz_gt, viz_path)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()