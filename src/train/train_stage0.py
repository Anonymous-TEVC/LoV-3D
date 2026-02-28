"""
Stage 0: Encoder Warmup via Multi-Task Volume Regression.

Trains a MONAI ResNet-50 (conv1->layer3) to predict ICV-normalized volumes
for 5 AD-signature regions from 128^3 T1-weighted MRI. The trained encoder
is frozen and used in all subsequent stages.

Regions: Hippocampus, Entorhinal, Fusiform, MidTemporal, Ventricles
Loss: SmoothL1 (Huber) on population z-score-normalized volumes
Input: baseline-only scans (no follow-ups)

Default training config:
  - Train: ~1127, Val: ~141 baseline scans
  - Epochs: 80, Batch: 16 (x2 accum = 32 effective)
  - LR: 1.4e-4, cosine schedule with 5-epoch linear warmup
  - Optimizer: AdamW (weight_decay=1e-4)
  - Parameters: backbone=46.2M, heads=2.6M, total=48.8M

Usage:
  python src/train/train_stage0.py \
      --data_json data_splits/stage0_data.json \
      --output_dir checkpoints/stage0
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as ckpt_fn
from monai.networks.nets import resnet50
from torch.amp import autocast


# --- Regions ----------------------------------------------------------------
REGIONS = ['Hippocampus', 'Entorhinal', 'Fusiform', 'MidTemp', 'Ventricles']
NUM_REGIONS = len(REGIONS)


# --- Dataset ----------------------------------------------------------------
class Stage0Dataset(Dataset):
    """Loads baseline-only MRI scans with FreeSurfer volume targets."""

    def __init__(self, samples, vol_mean, vol_std, augment=False):
        """
        Args:
            samples: list of dicts with keys 'npy_path' and 'volumes'
            vol_mean: dict of region -> mean (from training set)
            vol_std: dict of region -> std
            augment: whether to apply data augmentation (random flip + noise)
        """
        self.samples = samples
        self.vol_mean = vol_mean
        self.vol_std = vol_std
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        vol = np.load(s['npy_path']).astype(np.float32)

        if self.augment:
            if np.random.random() < 0.5:
                vol = np.flip(vol, axis=2).copy()
            vol = vol + np.random.normal(0, 0.02, vol.shape).astype(np.float32)
            vol = vol * np.random.uniform(0.95, 1.05)

        vol = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)

        # Population z-score normalization
        targets = []
        for r in REGIONS:
            v = s['volumes'][r]
            z = (v - self.vol_mean[r]) / (self.vol_std[r] + 1e-8)
            targets.append(z)
        targets = torch.tensor(targets, dtype=torch.float32)

        return vol, targets


# --- Model ------------------------------------------------------------------
class EncoderWithHeads(nn.Module):
    """ResNet-50 (conv1->layer3) + per-region regression heads."""

    def __init__(self, use_grad_ckpt=True, dropout=0.3):
        super().__init__()

        backbone = resnet50(
            pretrained=False, spatial_dims=3,
            n_input_channels=1, num_classes=1,
        )

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.act = backbone.act
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.use_grad_ckpt = use_grad_ckpt

        self.gap = nn.AdaptiveAvgPool3d(1)

        # Per-region regression heads (1024 -> 512 -> 1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 512, bias=False),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 1),
            )
            for _ in range(NUM_REGIONS)
        ])

    def _run_layer(self, layer, x):
        if self.use_grad_ckpt and self.training:
            return ckpt_fn(layer, x, use_reentrant=False)
        return layer(x)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 128, 128, 128)
        Returns:
            preds: (B, NUM_REGIONS) predicted z-scored volumes
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.maxpool(x)
        x = self._run_layer(self.layer1, x)
        x = self._run_layer(self.layer2, x)
        x = self._run_layer(self.layer3, x)  # (B, 1024, 16, 16, 16)

        feat = self.gap(x).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 1024)
        preds = torch.cat([head(feat) for head in self.heads], dim=-1)  # (B, 5)
        return preds

    def save_encoder(self, path):
        """Save only the encoder layers (conv1->layer3) for Stage 1."""
        state_dict = {}
        encoder_prefixes = ('conv1', 'bn1', 'act', 'maxpool', 'layer1', 'layer2', 'layer3')
        for name, param in self.named_parameters():
            if name.startswith(encoder_prefixes):
                state_dict[name] = param.data
        for name, buf in self.named_buffers():
            if name.startswith(encoder_prefixes):
                state_dict[name] = buf
        torch.save(state_dict, path)
        print(f"Saved encoder ({len(state_dict)} keys) to {path}")


# --- Evaluation -------------------------------------------------------------
def evaluate(model, loader, criterion, vol_mean, vol_std, device,
             loss_reduction='mean'):
    """Evaluate with MAE, Pearson r, 3-class accuracy, and per-class F1."""
    model.eval()
    total_loss = 0
    all_preds = {r: [] for r in REGIONS}
    all_targets = {r: [] for r in REGIONS}

    with torch.no_grad():
        for vol, targets in loader:
            vol, targets = vol.to(device), targets.to(device)
            with autocast('cuda', dtype=torch.bfloat16):
                preds = model(vol)
                if loss_reduction == 'mean':
                    loss = criterion(preds, targets)
                else:
                    loss = criterion(preds, targets).sum(dim=1).mean()
            total_loss += loss.item() * vol.size(0)

            for i, r in enumerate(REGIONS):
                all_preds[r].extend(preds[:, i].float().cpu().numpy())
                all_targets[r].extend(targets[:, i].float().cpu().numpy())

    n = len(loader.dataset)
    avg_loss = total_loss / n

    region_results = []
    for r in REGIONS:
        p = np.array(all_preds[r])
        t = np.array(all_targets[r])

        # De-normalize to raw ICV-normalized volume scale for MAE
        p_raw = p * vol_std[r] + vol_mean[r]
        t_raw = t * vol_std[r] + vol_mean[r]
        mae = np.abs(p_raw - t_raw).mean()
        corr = np.corrcoef(p, t)[0, 1] if len(p) > 1 else 0.0

        # Discretize to 3 classes for accuracy/F1
        if r == 'Ventricles':
            pred_cls = np.where(p < 0.5, 0, np.where(p < 1.5, 1, 2))
            true_cls = np.where(t < 0.5, 0, np.where(t < 1.5, 1, 2))
        else:
            pred_cls = np.where(p > -0.5, 0, np.where(p > -1.5, 1, 2))
            true_cls = np.where(t > -0.5, 0, np.where(t > -1.5, 1, 2))

        acc = (pred_cls == true_cls).mean()

        f1_per_class = []
        for c in range(3):
            tp = ((pred_cls == c) & (true_cls == c)).sum()
            fp = ((pred_cls == c) & (true_cls != c)).sum()
            fn = ((pred_cls != c) & (true_cls == c)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f1_per_class.append(f1)
        macro_f1 = np.mean(f1_per_class)
        region_results.append((r, mae, corr, acc, macro_f1, f1_per_class))

    avg_f1 = np.mean([rr[4] for rr in region_results])

    print(f"Loss: {avg_loss:.4f}, Avg Macro F1: {avg_f1:.4f}")
    for r, mae, corr, acc, macro_f1, f1_per_class in region_results:
        print(f"  {r:12s}: MAE={mae:.6f}, r={corr:.3f}, "
              f"Acc={acc:.3f}, F1={macro_f1:.3f} "
              f"[{f1_per_class[0]:.3f}, {f1_per_class[1]:.3f}, {f1_per_class[2]:.3f}]")

    return avg_loss, avg_f1


# --- Main -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Stage 0: Encoder Warmup')
    parser.add_argument('--data_json', type=str, required=True,
                        help='Path to JSON with train/val/test splits and FreeSurfer volumes')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save encoder checkpoint')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accum_steps', type=int, default=2,
                        help='Gradient accumulation steps (effective batch = batch_size x accum)')
    parser.add_argument('--lr', type=float, default=1.4e-4)
    parser.add_argument('--warmup_steps', type=int, default=175,
                        help='Linear warmup steps (default 175 = 5 epochs for ~1100 baseline scans)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm (0 = no clipping)')
    parser.add_argument('--loss_reduction', type=str, default='mean',
                        choices=['region_sum', 'mean'],
                        help='Loss reduction: mean = standard mean; region_sum = sum over 5 regions')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation (random flip + noise + intensity)')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate in regression heads')
    parser.add_argument('--seed', type=int, default=-1,
                        help='Random seed (-1 = random)')
    args = parser.parse_args()

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    # -- Load data --
    print("\n=== Loading data ===")
    with open(args.data_json) as f:
        data = json.load(f)

    train_samples = data['train']
    val_samples = data['val']
    test_samples = data.get('test', [])
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Compute volume statistics from training set
    print("\nVolume statistics (from train):")
    vol_mean = {r: 0.0 for r in REGIONS}
    vol_std = {r: 0.0 for r in REGIONS}
    for r in REGIONS:
        vals = [s['volumes'][r] for s in train_samples]
        vol_mean[r] = np.mean(vals)
        vol_std[r] = np.std(vals)
        print(f"  {r:12s}: mean={vol_mean[r]:.6f}, std={vol_std[r]:.6f}")

    train_ds = Stage0Dataset(train_samples, vol_mean, vol_std, augment=args.augment)
    val_ds = Stage0Dataset(val_samples, vol_mean, vol_std, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # -- Build model --
    print("\n=== Building model ===")
    model = EncoderWithHeads(use_grad_ckpt=True, dropout=args.dropout).to(device)
    print("  Gradient checkpointing enabled")
    if args.augment:
        print("  Data augmentation enabled (flip + noise + intensity)")

    backbone_params = sum(p.numel() for n, p in model.named_parameters()
                          if not n.startswith('heads'))
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith('heads'))
    print(f"Parameters: total={backbone_params + head_params:,}, "
          f"backbone={backbone_params:,}, heads={head_params:,}")

    if args.loss_reduction == 'mean':
        criterion = nn.SmoothL1Loss(reduction='mean')
    else:
        criterion = nn.SmoothL1Loss(reduction='none')
    print(f"Loss: SmoothL1Loss (Huber), reduction={args.loss_reduction}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=args.weight_decay)
    print(f"Optimizer: AdamW (wd={args.weight_decay})")
    print(f"LR: {args.lr:.1e}")

    batches_per_epoch = len(train_loader)
    opt_steps_per_epoch = batches_per_epoch // args.accum_steps
    total_steps = opt_steps_per_epoch * args.epochs
    print(f"Steps/epoch: {batches_per_epoch}, Effective batch: {args.batch_size * args.accum_steps}")
    print(f"Optimizer steps: {total_steps}, Warmup: {args.warmup_steps}")

    # Cosine schedule with linear warmup
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -- Training loop --
    print("\n=== Training ===")
    best_f1 = 0.0
    no_improve = 0
    global_step = 0

    import time

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, (vol, targets) in enumerate(train_loader, 1):
            vol, targets = vol.to(device), targets.to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                preds = model(vol)
                if args.loss_reduction == 'mean':
                    loss = criterion(preds, targets) / args.accum_steps
                else:
                    per_region = criterion(preds, targets)  # (B, 5)
                    loss = per_region.sum(dim=1).mean() / args.accum_steps

            loss.backward()
            epoch_loss += loss.item() * args.accum_steps

            if batch_idx % args.accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if batch_idx % 10 == 0:
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Epoch {epoch} Step {batch_idx}"
                          f"/{batches_per_epoch}: loss={loss.item() * args.accum_steps:.4f}, lr={lr:.6f}")

        elapsed = time.time() - t0
        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch}/{args.epochs} ({elapsed:.0f}s): train_loss={avg_loss:.4f}")

        # Validation
        print(f"  Val: ", end='', flush=True)
        val_loss, val_f1 = evaluate(model, val_loader, criterion, vol_mean, vol_std,
                                     device, args.loss_reduction)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            model.save_encoder(os.path.join(args.output_dir, 'encoder_layer3.pt'))
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'full_model.pt'))
            print(f"  * New best! F1={val_f1:.4f} (saved)")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # -- Final evaluation with best model --
    print("\n=== Final evaluation with best model ===")
    best_state = torch.load(os.path.join(args.output_dir, 'full_model.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_state)

    print("\nVal (best):")
    evaluate(model, val_loader, criterion, vol_mean, vol_std, device, args.loss_reduction)

    if test_samples:
        test_ds = Stage0Dataset(test_samples, vol_mean, vol_std)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True)
        print("\nTest:")
        evaluate(model, test_loader, criterion, vol_mean, vol_std, device, args.loss_reduction)

    print(f"\n=== Summary ===")
    print(f"Best epoch: {best_epoch}, Val F1: {best_f1:.4f}")
    print(f"Encoder: {os.path.join(args.output_dir, 'encoder_layer3.pt')}")


if __name__ == '__main__':
    main()
