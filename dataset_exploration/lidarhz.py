import h5py
import numpy as np

with h5py.File('dataset/Training-validation set (11:36 and 12:43)/frog_11-36_12-43_train_val.h5', 'r') as f:
    ts = f['timestamps'][:20]  # first 20 timestamps

diffs = np.diff(ts)
print("Time between scans (seconds):", diffs)
print("Estimated frequency (Hz):", 1.0 / diffs.mean())