# Script to understand more about the dataset in order to use it in future work.

"""
FROG Dataset Exploration Script
Step 1: Load, analyze, and visualize the FROG HDF5 dataset.
 
Usage:
    python explore_frog.py --file frog_11-36_12-43_train_val.h5
"""
 
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os
 
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LASER_MIN_ANGLE = -np.pi / 2   # -90 degrees (right side)
LASER_MAX_ANGLE =  np.pi / 2   #  90 degrees (left side)
LASER_NUM_POINTS = 720
MAX_RANGE = 10.0                # meters — FROG uses 10m max
NUM_SCANS_TO_VISUALIZE = 6      # how many scans to plot
 
 
# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
def load_frog(filepath):
    print(f"\n{'='*50}")
    print(f"Loading: {filepath}")
    print(f"{'='*50}")
 
    with h5py.File(filepath, 'r') as f:
        print(f"\nKeys in file: {list(f.keys())}")
 
        scans       = f['scans'][:]        # (N, 720) float32
        timestamps  = f['timestamps'][:]   # (N,)     float64
        circles     = f['circles'][:]      # (M, 6)   float32
        circle_idx  = f['circle_idx'][:]   # (N,)     uint32
        circle_num  = f['circle_num'][:]   # (N,)     uint32
 
        # split only exists in train/val file
        split = f['split'][:] if 'split' in f else None
 
    return scans, timestamps, circles, circle_idx, circle_num, split
 
 
# ─────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────
def print_statistics(scans, timestamps, circles, circle_idx, circle_num, split):
    N = len(scans)
    M = len(circles)
 
    print(f"\n{'='*50}")
    print("DATASET STATISTICS")
    print(f"{'='*50}")
 
    print(f"\n--- Scans ---")
    print(f"  Total scans (N):            {N:,}")
    print(f"  Points per scan:            {scans.shape[1]}")
 
    duration_sec = timestamps[-1] - timestamps[0]
    print(f"  Recording duration:         {duration_sec/60:.1f} minutes")
    avg_hz = N / duration_sec
    print(f"  Average scan frequency:     {avg_hz:.1f} Hz")
 
    valid_ranges = scans[(scans > 0) & (scans < MAX_RANGE)]
    print(f"  Mean valid range:           {valid_ranges.mean():.2f} m")
    print(f"  Median valid range:         {np.median(valid_ranges):.2f} m")
 
    print(f"\n--- People Annotations ---")
    print(f"  Total annotations (M):      {M:,}")
 
    people_per_scan = circle_num
    populated = people_per_scan > 0
    print(f"  Scans with people:          {populated.sum():,} ({100*populated.mean():.1f}%)")
    print(f"  Scans without people:       {(~populated).sum():,}")
    print(f"  Avg people per scan:        {people_per_scan.mean():.2f}")
    print(f"  Max people in one scan:     {people_per_scan.max()}")
 
    # circles columns: x, y, radius, distance, angle, angular_radius
    distances = circles[:, 3]
    print(f"\n--- Person Distances ---")
    print(f"  Mean distance to person:    {distances.mean():.2f} m")
    print(f"  Median distance:            {np.median(distances):.2f} m")
    print(f"  Min distance:               {distances.min():.2f} m")
    print(f"  Max distance:               {distances.max():.2f} m")
 
    if split is not None:
        train_count = (split == 0).sum()
        val_count   = (split == 1).sum()
        print(f"\n--- Train/Val Split ---")
        print(f"  Training scans:             {train_count:,} ({100*train_count/N:.1f}%)")
        print(f"  Validation scans:           {val_count:,} ({100*val_count/N:.1f}%)")
 
    print(f"\n--- Range Validity ---")
    inf_count = np.isinf(scans).sum()
    zero_count = (scans == 0).sum()
    print(f"  Inf readings:               {inf_count:,}")
    print(f"  Zero readings:              {zero_count:,}")
 
 
# ─────────────────────────────────────────────
# CONVERSION HELPERS
# ─────────────────────────────────────────────
def scan_to_cartesian(scan):
    """Convert a 1D range scan to (x, y) Cartesian points, filtering invalid readings."""
    angles = np.linspace(LASER_MIN_ANGLE, LASER_MAX_ANGLE, LASER_NUM_POINTS)
    valid  = (scan > 0) & (scan < MAX_RANGE) & np.isfinite(scan)
    x = scan[valid] * np.cos(angles[valid])
    y = scan[valid] * np.sin(angles[valid])
    return x, y
 
 
def get_annotations_for_scan(i, circles, circle_idx, circle_num):
    """Return the annotations (x, y, radius) for scan index i."""
    idx = circle_idx[i]
    num = circle_num[i]
    if num == 0:
        return []
    ann = circles[idx: idx + num]
    # columns: x, y, radius, distance, angle, angular_radius
    return [(row[0], row[1], row[2]) for row in ann]
 
 
# ─────────────────────────────────────────────
# DISTANCE HISTOGRAM
# ─────────────────────────────────────────────
def plot_distance_histogram(circles):
    distances = circles[:, 3]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(distances, bins=50, color='steelblue', edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Distance to person (m)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Person Detection Distances")
    ax.axvline(distances.mean(), color='red', linestyle='--', label=f"Mean: {distances.mean():.2f}m")
    ax.legend()
    plt.tight_layout()
    plt.savefig("frog_distance_histogram.png", dpi=150)
    print("\n  Saved: frog_distance_histogram.png")
    plt.show()
 
 
# ─────────────────────────────────────────────
# PEOPLE PER SCAN HISTOGRAM
# ─────────────────────────────────────────────
def plot_people_per_scan(circle_num):
    fig, ax = plt.subplots(figsize=(8, 4))
    max_people = circle_num.max()
    bins = np.arange(0, max_people + 2) - 0.5
    ax.hist(circle_num, bins=bins, color='coral', edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Number of people in scan")
    ax.set_ylabel("Number of scans")
    ax.set_title("Distribution of People Per Scan")
    ax.set_xticks(range(0, max_people + 1))
    plt.tight_layout()
    plt.savefig("frog_people_per_scan.png", dpi=150)
    print("  Saved: frog_people_per_scan.png")
    plt.show()
 
 
# ─────────────────────────────────────────────
# SCAN VISUALIZATIONS
# ─────────────────────────────────────────────
def visualize_scans(scans, circles, circle_idx, circle_num, num_scans=6):
    """
    Plot a grid of individual scans with person annotations overlaid.
    Picks scans that actually contain people for informative plots.
    """
    # find scans that have at least 1 person annotation
    populated_indices = np.where(circle_num > 0)[0]
 
    if len(populated_indices) == 0:
        print("  No annotated scans found to visualize.")
        return
 
    # sample evenly across the sequence so we see variety
    chosen = populated_indices[
        np.linspace(0, len(populated_indices) - 1, num_scans, dtype=int)
    ]
 
    cols = 3
    rows = int(np.ceil(num_scans / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
 
    for plot_idx, scan_idx in enumerate(chosen):
        ax = axes[plot_idx]
        scan = scans[scan_idx]
        x, y = scan_to_cartesian(scan)
        anns = get_annotations_for_scan(scan_idx, circles, circle_idx, circle_num)
 
        # plot laser points
        ax.scatter(x, y, s=2, c='steelblue', alpha=0.6, label='LiDAR points')
 
        # plot person annotation circles
        for (px, py, pr) in anns:
            circle = plt.Circle((px, py), pr, color='limegreen',
                                 fill=False, linewidth=2)
            ax.add_patch(circle)
            ax.plot(px, py, 'g+', markersize=8, markeredgewidth=2)
 
        ax.set_xlim(-MAX_RANGE, MAX_RANGE)
        ax.set_ylim(-MAX_RANGE, MAX_RANGE)
        ax.set_aspect('equal')
        ax.set_title(f"Scan #{scan_idx}  |  {len(anns)} person(s)", fontsize=9)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
 
        # robot position marker
        ax.plot(0, 0, 'r^', markersize=10, label='Robot')
 
        if plot_idx == 0:
            ax.legend(loc='upper right', fontsize=7)
 
    # hide any unused subplots
    for i in range(num_scans, len(axes)):
        axes[i].set_visible(False)
 
    plt.suptitle("FROG Dataset — Sample Annotated Scans\n"
                 "(Green circles = annotated people, Blue dots = LiDAR points)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("frog_sample_scans.png", dpi=150, bbox_inches='tight')
    print("  Saved: frog_sample_scans.png")
    plt.show()
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Explore the FROG HDF5 dataset.")
    parser.add_argument(
        '--file',
        type=str,
        default='dataset/Training-validation set (11:36 and 12:43)/frog_11-36_12-43_train_val.h5',
        help='Path to the FROG HDF5 file'
    )
    args = parser.parse_args()
 
    if not os.path.exists(args.file):
        print(f"\nERROR: File not found: {args.file}")
        print("Make sure the HDF5 file is in the same directory, or pass --file <path>")
        return
 
    # 1. Load
    scans, timestamps, circles, circle_idx, circle_num, split = load_frog(args.file)
 
    # 2. Print statistics
    print_statistics(scans, timestamps, circles, circle_idx, circle_num, split)
 
    # 3. Plots
    print(f"\n{'='*50}")
    print("GENERATING PLOTS")
    print(f"{'='*50}\n")
 
    plot_distance_histogram(circles)
    plot_people_per_scan(circle_num)
    visualize_scans(scans, circles, circle_idx, circle_num,
                    num_scans=NUM_SCANS_TO_VISUALIZE)
 
    print(f"\n{'='*50}")
    print("Done! Check the saved .png files in your working directory.")
    print(f"{'='*50}\n")
 
 
if __name__ == '__main__':
    main()