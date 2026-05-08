"""
frog_dataset.py
PyTorch Dataset for the FROG HDF5 laser scan people detection dataset.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# Constants matching FROG sensor specs
# ─────────────────────────────────────────────
LASER_NUM_POINTS = 720
MAX_RANGE = 61.0   # meters
PERSON_RADIUS    = 0.5    # meters — used for segmentation target


class FrogDataset(Dataset):
    """
    Loads FROG HDF5 data and returns (scan, segmentation_mask) pairs.

    The segmentation mask is a binary vector of length 720:
        1 = this laser point belongs to a person
        0 = background

    Args:
        filepath  : path to the HDF5 file
        split     : 'train', 'val', or 'test'
                    For the test file there is no split array — pass 'test'.
        augment   : whether to apply data augmentation
    """

    def __init__(self, filepath, split='train', augment=False):
        super().__init__()
        self.augment = augment

        with h5py.File(filepath, 'r') as f:
            scans      = f['scans'][:]       # (N, 720)
            circles    = f['circles'][:]     # (M, 6)
            circle_idx = f['circle_idx'][:]  # (N,)
            circle_num = f['circle_num'][:]  # (N,)

            if split in ('train', 'val'):
                split_arr = f['split'][:]    # (N,)  0=train 1=val
                mask = (split_arr == 0) if split == 'train' else (split_arr == 1)
                indices = np.where(mask)[0]
            else:
                # test file has no split array — use everything
                indices = np.arange(len(scans))

        # Only keep scans that have at least one person annotation
        populated = circle_num[indices] > 0
        indices   = indices[populated]

        self.original_indices = indices
        self.scans      = scans[indices]           # (N', 720)
        self.circle_idx = circle_idx[indices]      # (N',)
        self.circle_num = circle_num[indices]      # (N',)
        self.circles    = circles                  # full array — indexed per scan

        print(f"[FrogDataset] split='{split}'  scans={len(self.scans):,}")

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        scan = self.scans[i].copy()   # (720,)

        # ── Preprocessing ──────────────────────────────
        # Clip to max range and replace invalid readings
        scan = np.where(np.isfinite(scan), scan, 61.0)
        scan = np.clip(scan, 0.0, 61.0)
        scan = scan / 61.0

        # ── Build segmentation target ──────────────────
        target = self._build_segmentation_mask(i, scan * MAX_RANGE)

        # ── Augmentation ───────────────────────────────
        if self.augment:
            scan, target = self._augment(scan, target)

        # Shape: (1, 720) — channel-first for Conv1d
        scan_t   = torch.tensor(scan,   dtype=torch.float32).unsqueeze(0)
        target_t = torch.tensor(target, dtype=torch.float32)

        return scan_t, target_t

    # ──────────────────────────────────────────────────
    def _build_segmentation_mask(self, i, raw_scan):
        """
        For each laser point, check whether it falls inside any
        person annotation circle. Returns a binary vector (720,).
        """
        mask       = np.zeros(LASER_NUM_POINTS, dtype=np.float32)
        angles     = np.linspace(-np.pi / 2, np.pi / 2, LASER_NUM_POINTS)

        idx = self.circle_idx[i]
        num = self.circle_num[i]
        if num == 0:
            return mask

        # Cartesian coordinates of each laser point
        xs = raw_scan * np.cos(angles)   # (720,)
        ys = raw_scan * np.sin(angles)   # (720,)

        for ann in self.circles[idx: idx + num]:
            px, py, pr = ann[0], ann[1], ann[2]
            dist_sq = (xs - px) ** 2 + (ys - py) ** 2
            mask[dist_sq <= pr ** 2] = 1.0

        return mask

    # ──────────────────────────────────────────────────
    def _augment(self, scan, target):
        """Simple augmentation: random horizontal flip."""
        if np.random.rand() < 0.5:
            scan   = scan[::-1].copy()
            target = target[::-1].copy()
        return scan, target