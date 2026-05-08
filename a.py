import numpy as np
import h5py


with h5py.File('dataset/Training-validation set (11:36 and 12:43)/frog_11-36_12-43_train_val.h5', 'r') as f:
    train_scan = f['scans'][0][:]

print("\nTrain scan:")
print("  Has inf:", np.any(np.isinf(train_scan)))
print("  min/max:", train_scan.min(), train_scan.max())
print("  zeros:", (train_scan == 0).sum())
print("  sample:", train_scan[:10])