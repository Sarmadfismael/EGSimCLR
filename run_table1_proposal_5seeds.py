"""
Table I, "Proposed method" block, 5-seed run (GRSL rebuttal, R1.4: statistical robustness).

Table I in the paper reports the proposed entropy-guided method fine-tuned on
MSD after pre-training on three different source datasets (AID, EuroSAT, UC
Merced) -- three rows, each currently a single point estimate:

    Proposed method   AID         98.52  98.61  98.52  98.49
                       EuroSAT     96.30  96.56  96.30  96.29
                       UC Merced   91.85  92.76  91.85  91.84

This script reruns all three rows across 5 seeds each (15 fine-tuning runs
total) on the FIXED split_dataset1 (MSD), and reports mean +/- std per
pretraining dataset so Table I can cite a defensible statistic instead of a
single 15-images/class test-set number.

Optimized for turnaround, not tidiness:
  - mixed precision (torch.cuda.amp)
  - multi-worker, pinned-memory DataLoaders
  - best checkpoint kept in memory during training (no per-epoch disk I/O)
  - no per-epoch console spam / no confusion-matrix or curve plots
  - one line printed per (pretrain-dataset, seed) run, one summary table at the end

Usage:
    python run_table1_proposal_5seeds.py
"""

import os
import copy
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import config
from models.finetuneSimCLR_proposal_copy import get_model
from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1

# --------------------------------------------------------------------------- #
# EDIT ME: one proposal-method checkpoint per pre-training dataset
# Paths derive from config.GRSL_ROOT (env var GRSL_ROOT overrides the
# Windows default) so moving machines only requires setting one env var.
CODE_ROOT = os.path.join(config.GRSL_ROOT, "code/sc")
PRETRAIN_CHECKPOINTS = {
    "AID": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug20_02-40-55_DESKTOP-SU0KM9L_AID/checkpoint_0200.pth.tar"),
    "EuroSAT": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug21_03-07-29_DESKTOP-SU0KM9L_EuroSAT/checkpoint_0200.pth.tar"),
    "UC_Merced": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug25_01-20-02_DESKTOP-SU0KM9L_Uc_Merced/checkpoint_0200.pth.tar"),
}
SEEDS = [42, 123, 456, 789, 1011]
DATA_DIR = config.DATA_DIR                      # fixed split_dataset1 (MSD); only the seed varies
RESULTS_DIR = "results_table1_proposal_5seeds"
NUM_WORKERS = 0                                 # 0 avoids Windows multiprocessing/pagefile crashes; raise if your system handles it
SAVE_PER_RUN_CHECKPOINT = True                  # set False to skip disk writes entirely (fastest, no reuse later)
# --------------------------------------------------------------------------- #

USE_AMP = torch.cuda.is_available()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(data_dir, seed):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, generator=g,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=config.BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return train_loader, val_loader, test_loader, train_set.classes


def run_one(checkpoint_path, seed):
    set_seed(seed)
    train_loader, val_loader, test_loader, class_names = build_loaders(DATA_DIR, seed)

    model = get_model(checkpoint_path=checkpoint_path).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0
    best_epoch = -1

    for epoch in range(config.EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- validation (for early stopping / best-epoch selection only) ---
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(config.DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    outputs = model(inputs)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = compute_accuracy(val_labels, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"    epoch {epoch+1:>3}/{config.EPOCHS}  val_acc={val_acc*100:5.2f}%  "
            f"best={best_val_acc*100:5.2f}%  no_improve={epochs_no_improve:>2}/{config.patience}",
            end="\r", flush=True,
        )

        if epochs_no_improve >= config.patience:
            print()  # move past the \r line before the early-stop break
            break

    print()  # move past the \r line once training finishes normally too
    # --- final test evaluation using the best validation checkpoint ---
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(inputs)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_labels.extend(labels.numpy())

    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_trained": epoch + 1,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": compute_accuracy(test_labels, test_preds),
        "test_precision": compute_precision(test_labels, test_preds),
        "test_recall": compute_recall(test_labels, test_preds),
        "test_f1": compute_f1(test_labels, test_preds),
    }
    return metrics, best_state


def main():
    for name, path in PRETRAIN_CHECKPOINTS.items():
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found: {path}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    per_dataset_metrics = {}  # name -> list of per-seed metric dicts
    t0 = time.time()

    for name, checkpoint_path in PRETRAIN_CHECKPOINTS.items():
        print(f"\n{'='*70}\nPROPOSED METHOD, pretrained on {name}\n{'='*70}")
        dataset_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        run_metrics = []
        for seed in SEEDS:
            t_run = time.time()
            m, best_state = run_one(checkpoint_path, seed)
            run_metrics.append(m)

            if SAVE_PER_RUN_CHECKPOINT:
                torch.save(best_state, os.path.join(dataset_dir, f"best_model_seed{seed}.pth"))

            print(
                f"  [{name} | seed {seed:>5}] acc={m['test_accuracy']*100:6.2f}%  "
                f"prec={m['test_precision']*100:6.2f}%  rec={m['test_recall']*100:6.2f}%  "
                f"f1={m['test_f1']*100:6.2f}%  "
                f"(best epoch {m['best_epoch']+1}/{m['epochs_trained']}, {time.time()-t_run:5.1f}s)"
            )

        per_dataset_metrics[name] = run_metrics

        with open(os.path.join(dataset_dir, "per_seed_results.json"), "w") as f:
            json.dump(run_metrics, f, indent=2)

    # --- aggregate mean +/- std per pretraining dataset (i.e. per Table I row) ---
    table_rows = []
    for name, run_metrics in per_dataset_metrics.items():
        accs = np.array([m["test_accuracy"] for m in run_metrics])
        precs = np.array([m["test_precision"] for m in run_metrics])
        recs = np.array([m["test_recall"] for m in run_metrics])
        f1s = np.array([m["test_f1"] for m in run_metrics])

        table_rows.append({
            "pretraining_dataset": name,
            "n_seeds": len(run_metrics),
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "precision_mean": float(precs.mean()), "precision_std": float(precs.std()),
            "recall_mean": float(recs.mean()), "recall_std": float(recs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        })

    summary_path = os.path.join(RESULTS_DIR, "table1_proposed_summary.json")
    with open(summary_path, "w") as f:
        json.dump(table_rows, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTable I - Proposed method (mean +/- std over {len(SEEDS)} seeds, %)\n{'='*70}")
    print(f"{'Pretraining':<12} {'Acc.':>14} {'Prec.':>14} {'Recall':>14} {'F1':>14}")
    for row in table_rows:
        print(
            f"{row['pretraining_dataset']:<12} "
            f"{row['accuracy_mean']*100:6.2f}+/-{row['accuracy_std']*100:<5.2f} "
            f"{row['precision_mean']*100:6.2f}+/-{row['precision_std']*100:<5.2f} "
            f"{row['recall_mean']*100:6.2f}+/-{row['recall_std']*100:<5.2f} "
            f"{row['f1_mean']*100:6.2f}+/-{row['f1_std']*100:<5.2f}"
        )
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Saved: {summary_path}")
    print(f"Per-seed results under: {RESULTS_DIR}/<dataset>/per_seed_results.json")


if __name__ == "__main__":
    main()
