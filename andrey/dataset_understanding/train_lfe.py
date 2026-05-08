"""
train_lfe.py
Trains the LFE segmentation model on the FROG dataset.

Usage:
    python train_lfe.py --train_file path/to/frog_11-36_12-43_train_val.h5

Outputs (saved to ./checkpoints/):
    best_lfe.pt        — best model weights by validation loss
    last_lfe.pt        — weights after final epoch
    training_log.csv   — loss/F1 per epoch for plotting
"""

import os
import csv
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from andrey.dataset_understanding.frog_dataset import FrogDataset
from andrey.dataset_understanding.lfe_model    import LFESegmentation


# ─────────────────────────────────────────────
# Config — edit these as needed
# ─────────────────────────────────────────────
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 4e-3
MAX_EPOCHS    = 100
PATIENCE      = 20          # early stopping patience
MIN_DELTA     = 1e-3        # minimum improvement to reset patience
NUM_WORKERS   = 4           # dataloader workers
CHECKPOINT_DIR = 'checkpoints'


# ─────────────────────────────────────────────
# Loss: BCE + Dice (mixed loss from paper)
# ─────────────────────────────────────────────

def dice_loss(probs, targets, eps=1e-6):
    """Soft Dice loss for binary segmentation."""
    intersection = (probs * targets).sum(dim=-1)
    union        = probs.sum(dim=-1) + targets.sum(dim=-1)
    dice         = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def mixed_loss(logits, targets):
    """Average of BCE loss and Dice loss, as used in the FROG paper."""
    probs    = torch.sigmoid(logits)
    bce      = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dice     = dice_loss(probs, targets)
    return (bce + dice) / 2


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def compute_f1(logits, targets, threshold=0.5):
    """Per-batch binary F1 score."""
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return f1


# ─────────────────────────────────────────────
# Train / validate one epoch
# ─────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, training=True):
    model.train() if training else model.eval()
    total_loss = 0.0
    total_f1   = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for scans, targets in loader:
            scans   = scans.to(device)      # (B, 1, 720)
            targets = targets.to(device)    # (B, 720)

            logits = model(scans)           # (B, 720)
            loss   = mixed_loss(logits, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_f1   += compute_f1(logits, targets)
            n_batches  += 1

    return total_loss / n_batches, total_f1 / n_batches


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str,
                        default='dataset/frog_11-36_12-43_train_val.h5',
                        help='Path to FROG train/val HDF5 file')
    args = parser.parse_args()

    # ── Device ───────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ─────────────────────────────
    print("\nLoading datasets...")
    train_ds = FrogDataset(args.train_file, split='train', augment=True)
    val_ds   = FrogDataset(args.train_file, split='val',   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    print(f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ─────────────────────────────────
    model     = LFESegmentation().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Checkpointing ─────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    log_path = os.path.join(CHECKPOINT_DIR, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_f1',
                                  'val_loss',   'val_f1',  'lr'])

    # ── Training loop ─────────────────────────
    best_val_loss  = float('inf')
    patience_count = 0

    print(f"\n{'='*60}")
    print(f"Starting training for up to {MAX_EPOCHS} epochs")
    print(f"Early stopping patience: {PATIENCE} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_f1 = run_epoch(model, train_loader,
                                         optimizer, device, training=True)
        val_loss,   val_f1   = run_epoch(model, val_loader,
                                         optimizer, device, training=False)
        scheduler.step()
        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train loss: {train_loss:.4f}  f1: {train_f1:.4f} | "
              f"val loss: {val_loss:.4f}  f1: {val_f1:.4f} | "
              f"lr: {lr:.2e} | {elapsed:.1f}s")

        # Log
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_f1,
                                    val_loss,   val_f1,  lr])

        # Save best
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, 'best_lfe.pt'))
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # Save final weights regardless
    torch.save(model.state_dict(),
               os.path.join(CHECKPOINT_DIR, 'last_lfe.pt'))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Weights saved to: {CHECKPOINT_DIR}/")
    print(f"Training log:     {log_path}")


if __name__ == '__main__':
    main()