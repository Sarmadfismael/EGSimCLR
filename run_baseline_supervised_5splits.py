"""
Table I rebuild, Supervised (from-scratch) baseline, 5-split robustness
protocol (same treatment as the proposed method, so the comparison in
Table I is apples-to-apples).

Trains ResNet-50 from random initialization (no pretraining) directly on
each of the 5 independently sampled MS_Dataset splits (split_dataset1..5),
seed fixed at 42 so only the split varies. Single row (no pretraining
dataset dimension), so 5 runs total.

Resumable: if interrupted (crash/power loss), rerun the same command and it
will skip any split whose result is already saved.

Usage:
    python run_baseline_supervised_5splits.py
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
from models.resnet50 import get_model
from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1

# --------------------------------------------------------------------------- #
SEED = 42
BASE_DIR = config.MS_DATASET_BASE_DIR
SPLIT_NAMES = config.MS_DATASET_SPLIT_NAMES
RESULTS_DIR = "results_table1_supervised_5splits"
NUM_WORKERS = 0
SAVE_PER_RUN_CHECKPOINT = True
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


def run_one(data_dir):
    set_seed(SEED)
    train_loader, val_loader, test_loader, class_names = build_loaders(data_dir, SEED)

    model = get_model().to(config.DEVICE)
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
            print()
            break

    print()
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
    for split_name in SPLIT_NAMES:
        data_dir = os.path.join(BASE_DIR, split_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Split folder not found: {data_dir}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    results_path = os.path.join(RESULTS_DIR, "per_split_results.json")
    split_metrics = []
    if os.path.isfile(results_path):
        with open(results_path, "r") as f:
            split_metrics = json.load(f)
    completed_splits = {m["split"] for m in split_metrics}

    t0 = time.time()
    print(f"\n{'='*70}\nSUPERVISED BASELINE (from scratch), seed={SEED}\n{'='*70}")

    for split_name in SPLIT_NAMES:
        if split_name in completed_splits:
            print(f"  [{split_name:>14}] already completed -- skipping (resume)")
            continue

        data_dir = os.path.join(BASE_DIR, split_name)
        t_split = time.time()
        m, best_state = run_one(data_dir)
        m["split"] = split_name
        split_metrics.append(m)

        if SAVE_PER_RUN_CHECKPOINT:
            torch.save(best_state, os.path.join(RESULTS_DIR, f"best_model_{split_name}.pth"))

        print(
            f"  [{split_name:>14}] acc={m['test_accuracy']*100:6.2f}%  "
            f"prec={m['test_precision']*100:6.2f}%  rec={m['test_recall']*100:6.2f}%  "
            f"f1={m['test_f1']*100:6.2f}%  "
            f"(best epoch {m['best_epoch']+1}/{m['epochs_trained']}, {time.time()-t_split:5.1f}s)"
        )

        with open(results_path, "w") as f:
            json.dump(split_metrics, f, indent=2)

    accs = np.array([m["test_accuracy"] for m in split_metrics])
    precs = np.array([m["test_precision"] for m in split_metrics])
    recs = np.array([m["test_recall"] for m in split_metrics])
    f1s = np.array([m["test_f1"] for m in split_metrics])

    summary = {
        "n_splits": len(split_metrics),
        "seed": SEED,
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "precision_mean": float(precs.mean()), "precision_std": float(precs.std()),
        "recall_mean": float(recs.mean()), "recall_std": float(recs.std()),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
    }

    summary_path = os.path.join(RESULTS_DIR, "table1_supervised_5splits_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTable I - Supervised baseline (mean +/- std over {len(SPLIT_NAMES)} splits, seed={SEED}, %)\n{'='*70}")
    print(
        f"Acc {summary['accuracy_mean']*100:.2f}+/-{summary['accuracy_std']*100:.2f}  "
        f"Prec {summary['precision_mean']*100:.2f}+/-{summary['precision_std']*100:.2f}  "
        f"Recall {summary['recall_mean']*100:.2f}+/-{summary['recall_std']*100:.2f}  "
        f"F1 {summary['f1_mean']*100:.2f}+/-{summary['f1_std']*100:.2f}"
    )
    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
