"""
R1 Major Comment 8 (w_ij item): lightweight final-checkpoint w_ij snapshot.

The existing pre-training runs did NOT log per-iteration w_ij statistics, and
re-running the full 200-epoch pre-training just to add that logging would risk
invalidating the just-finalized Table~I checkpoints under the revision
deadline. Instead of declining this item entirely, this script loads the
ALREADY-FINISHED final pre-training checkpoint and does a small number of
forward passes (no training, no gradient updates) over real AID images using
the identical two-view augmentation pipeline used during pre-training, then
computes the same q_ij / w_ij formula used in simclr.py::info_nce_loss at the
lambda value the schedule converges to by the final epoch (lambda ~= lambda_min
= 1.0). This gives an honest end-of-training weighting snapshot -- not a
full training-trajectory plot -- at effectively zero cost (a few seconds of
inference, no re-pretraining).

Formula (must match simclr.py::info_nce_loss exactly):
    sim_norm = (1 + cosine_similarity(z_i, z_j)) / 2
    H_avg    = (H_i + H_j) / 2        # normalized predictive entropy, avg of pair
    delta_H  = |H_i - H_j|
    q_ij     = sim_norm * (1 - H_avg) * exp(-delta_H)
    w_ij     = exp(-lambda * q_ij)                      # applied to NEGATIVE pairs only

lambda schedule (must match simclr.py __init__/train, lambda_min=1.0,
lambda_max=5.0, warmup_epochs=20):
    epoch < warmup:  lambda = lambda_max * (epoch+1) / warmup
    else:            progress = epoch / total_epochs
                      lambda = lambda_min + 0.5*(lambda_max-lambda_min)*(1+cos(pi*progress))

Usage (run wherever the final AID pre-training checkpoint lives -- the
authoritative one is batch=256, k_hat=32, per the fixed run.py used for
Table I -- and where the AID image folder is reachable):
    python w_ij_final_snapshot.py \
        --checkpoint /path/to/checkpoint_0200.pth.tar \
        --data /path/to/datasets_root \
        --dataset-name aid \
        --n-batches 10 \
        --output-dir results_r1_wij_snapshot
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from models.resnet_simclr_proposal import ResNetSimCLR

LAMBDA_MIN = 1.0
LAMBDA_MAX = 5.0
WARMUP_EPOCHS = 20


# ---------------------------------------------------------------------------
# Exact copy of entropy_simclr_full_project/data_aug/gaussian_blur.py, so this
# script has no cross-project import dependency and can be dropped next to
# either codebase.
# ---------------------------------------------------------------------------
class GaussianBlur(object):
    """Blur a single image on CPU (verbatim port of the pre-training augmentation)."""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                 stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                 stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias
        self.blur = nn.Sequential(nn.ReflectionPad2d(radias), self.blur_h, self.blur_v)
        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()
        return self.tensor_to_pil(img)


def simclr_transform(size=224, s=1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * size)),
        transforms.ToTensor(),
    ])


class TwoViewDataset(torch.utils.data.Dataset):
    """Wraps ImageFolder, returns two independently-augmented views per image (n_views=2)."""

    def __init__(self, root):
        self.base = ImageFolder(root=root)
        self.transform = simclr_transform()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.transform(img), self.transform(img)


def final_lambda(final_epoch, total_epochs):
    if final_epoch < WARMUP_EPOCHS:
        return LAMBDA_MAX * (final_epoch + 1) / WARMUP_EPOCHS
    progress = final_epoch / total_epochs
    return LAMBDA_MIN + 0.5 * (LAMBDA_MAX - LAMBDA_MIN) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def compute_batch_weights(model, view1, view2, lam, device):
    """Mirrors simclr.py::info_nce_loss's q_ij/w_ij computation for one batch."""
    images = torch.cat([view1, view2], dim=0).to(device)
    features, logits = model(images)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    entropy = -torch.sum(probs * log_probs, dim=1)
    max_entropy = torch.log(torch.tensor(logits.size(1), device=device, dtype=torch.float))
    norm_entropy = entropy / max_entropy

    H_i = norm_entropy.unsqueeze(1)
    H_j = norm_entropy.unsqueeze(0)
    H_avg = (H_i + H_j) / 2.0
    delta_H = torch.abs(H_i - H_j)

    sim_norm = (1.0 + similarity_matrix) / 2.0
    q_ij = sim_norm * (1.0 - H_avg) * torch.exp(-delta_H)
    w_ij = torch.exp(-lam * q_ij)

    n = images.size(0)
    batch_size = view1.size(0)
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(device)
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    self_mask = torch.eye(n, device=device).bool()
    mask[self_mask] = 0.0
    logits_mask = torch.ones_like(mask) - torch.eye(n, device=device)
    negative_mask = (~mask.bool()) & logits_mask.bool()

    return (
        q_ij[negative_mask].cpu().numpy(),
        w_ij[negative_mask].cpu().numpy(),
        norm_entropy.cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description="Lightweight final-checkpoint w_ij snapshot (R1 Major Comment 8)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_0200.pth.tar (final AID pre-training checkpoint)")
    parser.add_argument("--data", required=True, help="Root folder containing aid/uc_merced/EuroSAT_RGB (same as pre-training's -data)")
    parser.add_argument("--dataset-name", default="aid")
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--k-hat", type=int, default=32, help="Fixed K_HAT used for the authoritative Table I pre-training run")
    parser.add_argument("--batch-size", type=int, default=256, help="Authoritative Table I pre-training batch size")
    parser.add_argument("--n-batches", type=int, default=10, help="Number of batches to sample for the snapshot (lightweight -- not the full dataset)")
    parser.add_argument("--total-epochs", type=int, default=200)
    parser.add_argument("--final-epoch", type=int, default=199, help="0-indexed final completed epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results_r1_wij_snapshot")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    print(f"Device: {device}")

    lam = final_lambda(args.final_epoch, args.total_epochs)
    print(f"Final-epoch lambda (epoch {args.final_epoch}/{args.total_epochs}): {lam:.6f} "
          f"(schedule converges to lambda_min={LAMBDA_MIN})")

    model = ResNetSimCLR(base_model="resnet50", out_dim=args.out_dim, k_hat=args.k_hat).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} "
          f"(epoch in checkpoint: {checkpoint.get('epoch', 'n/a')})")

    data_dir = os.path.join(args.data, args.dataset_name)
    dataset = TwoViewDataset(data_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                          num_workers=2, drop_last=True)
    print(f"Dataset: {data_dir} ({len(dataset)} images) -- sampling {args.n_batches} batches "
          f"of size {args.batch_size}")

    all_q, all_w, all_h = [], [], []
    for i, (view1, view2) in enumerate(loader):
        if i >= args.n_batches:
            break
        q, w, h = compute_batch_weights(model, view1, view2, lam, device)
        all_q.append(q)
        all_w.append(w)
        all_h.append(h)
        print(f"  batch {i+1}/{args.n_batches}: mean q_ij={q.mean():.4f}  mean w_ij={w.mean():.4f}")

    all_q = np.concatenate(all_q)
    all_w = np.concatenate(all_w)
    all_h = np.concatenate(all_h)

    def stats(x):
        return {
            "mean": float(np.mean(x)), "std": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x)),
            "p10": float(np.percentile(x, 10)), "p50": float(np.percentile(x, 50)),
            "p90": float(np.percentile(x, 90)),
        }

    summary = {
        "final_lambda": lam,
        "final_epoch": args.final_epoch,
        "total_epochs": args.total_epochs,
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset_name,
        "batch_size": args.batch_size,
        "n_batches": args.n_batches,
        "n_negative_pairs_sampled": int(all_w.size),
        "q_ij": stats(all_q),
        "w_ij": stats(all_w),
        "normalized_entropy": stats(all_h),
    }

    print("\n=== Final-checkpoint w_ij snapshot ===")
    print(json.dumps(summary, indent=2))

    with open(os.path.join(args.output_dir, "wij_final_snapshot.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {os.path.join(args.output_dir, 'wij_final_snapshot.json')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(all_q, bins=50, color="steelblue")
        axes[0].set_title("q_ij (collision score)")
        axes[0].set_xlabel("q_ij")
        axes[0].set_ylabel("Number of negative pairs")
        axes[1].hist(all_w, bins=50, color="indianred")
        axes[1].set_title(f"w_ij (final lambda={lam:.3f})")
        axes[1].set_xlabel("w_ij")
        axes[1].set_ylabel("Number of negative pairs")
        fig.suptitle(f"Final-checkpoint w_ij snapshot ({args.dataset_name}, epoch {args.final_epoch})")
        fig.tight_layout()
        hist_path = os.path.join(args.output_dir, "wij_final_snapshot_hist.pdf")
        fig.savefig(hist_path)
        print(f"Saved: {hist_path}")
    except ImportError:
        print("matplotlib not available -- skipped histogram plot (JSON stats already saved).")


if __name__ == "__main__":
    main()
