"""
Table I rebuild, CMID baseline, 5-split robustness protocol (same treatment as
every other baseline / the proposed method, so the comparison is apples-to-
apples).

Fine-tunes the CMID-pretrained ResNet-50 (models/finetune_cmid.py -- loads
only the student encoder, per the CMID paper's Section III-A: "only the
student is used for downstream tasks") across the 5 independently sampled
MS_Dataset splits (split_dataset1..5), for each of the three pretraining
source datasets (AID, EuroSAT, UC Merced), seed fixed at 42 so only the split
varies.

Meant to run directly on the lab PC where CMID pretraining happened -- no
checkpoint download/transfer needed. Checkpoint discovery is automatic: this
script scans CMID_RUNS_DIR for run folders, reads each folder's config.yml
(written by CMID_project/utils.py's save_config_file) to find its
-dataset-name, and for each of the three datasets picks the most recently
modified checkpoint_0200.pth.tar among completed runs. If a dataset's
pretraining run is missing or incomplete, this fails loudly with exactly
what it found/didn't find -- it will not silently skip a dataset.

15 fine-tuning runs total (3 pretraining datasets x 5 splits). Resumable: if
interrupted, rerun the same command -- it skips any split whose result is
already saved.

Setup (lab PC):
    export GRSL_ROOT=/home/erhan/SarmadFolder      # parent of the 5-split dataset
    # optional, only if CMID_project's runs/ folder isn't at
    # $GRSL_ROOT/pkg/CMID_project/runs (it should be, per the user-reported
    # pretraining output location):
    export CMID_RUNS_DIR=/home/erhan/SarmadFolder/pkg/CMID_project/runs
    pip install scikit-learn pyyaml   # if not already in the venv

Usage:
    python run_baseline_cmid_5splits.py
"""

import os
import copy
import glob
import json
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import config
from models.finetune_cmid import get_model
from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1

# --------------------------------------------------------------------------- #
# CMID_project/run.py's `-dataset-name` choices are aid / EuroSAT_RGB /
# uc_merced -- map those to this project's {"AID", "EuroSAT", "UC_Merced"}
# labels so Table I stays consistent with every other baseline's naming.
DATASET_NAME_MAP = {
    "aid": "AID",
    "EuroSAT_RGB": "EuroSAT",
    "uc_merced": "UC_Merced",
}
PRETRAIN_EPOCH = 200  # must match CMID_project/run.py's --epochs default

# Where CMID_project's tensorboard SummaryWriter() wrote its runs/<...> dirs.
# Default matches the location the user reported after lab-PC pretraining
# completed: ~/SarmadFolder/pkg/CMID_project/runs. Override with env var
# CMID_RUNS_DIR if it's actually somewhere else on this machine.
CMID_RUNS_DIR = os.environ.get(
    "CMID_RUNS_DIR",
    os.path.join(config.GRSL_ROOT, "pkg/CMID_project/runs"),
)

SEED = 42
SPLIT_NAMES = config.MS_DATASET_SPLIT_NAMES
RESULTS_DIR = "results_table1_cmid_5splits"
NUM_WORKERS = 0
SAVE_PER_RUN_CHECKPOINT = True
# --------------------------------------------------------------------------- #

USE_AMP = torch.cuda.is_available()


def resolve_ms_dataset_base_dir():
    """config.MS_DATASET_BASE_DIR assumes GRSL_ROOT/dataset/MS_Dataset. If
    that's not where the split folders actually landed on this machine, fall
    back to searching under GRSL_ROOT for a directory that directly contains
    all 5 split_dataset folders, rather than failing on a path guess."""
    candidate = config.MS_DATASET_BASE_DIR
    if all(os.path.isdir(os.path.join(candidate, s)) for s in SPLIT_NAMES):
        return candidate

    for dirpath, dirnames, _ in os.walk(config.GRSL_ROOT):
        if all(s in dirnames for s in SPLIT_NAMES):
            return dirpath

    raise FileNotFoundError(
        f"Could not find split_dataset1..5 under GRSL_ROOT={config.GRSL_ROOT}\n"
        f"Set GRSL_ROOT to the correct parent directory, e.g.:\n"
        f"  export GRSL_ROOT=/home/erhan/SarmadFolder"
    )


def discover_cmid_checkpoints(runs_dir, epoch=PRETRAIN_EPOCH):
    """Scan runs_dir/*/config.yml for -dataset-name, matched against a
    checkpoint_<epoch>.pth.tar in the same folder. If a dataset has more than
    one completed run (e.g. a rerun/restart), the most recently modified
    checkpoint wins -- same "trust the newest completed run" rule already
    used elsewhere in this project when multiple candidate run folders showed
    up for one pretraining job."""
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(
            f"CMID_RUNS_DIR not found: {runs_dir}\n"
            f"Set the env var CMID_RUNS_DIR to wherever CMID_project's runs/ "
            f"folder actually lives, e.g.:\n"
            f"  export CMID_RUNS_DIR=/home/erhan/SarmadFolder/pkg/CMID_project/runs"
        )

    found = {}  # dataset_label -> (mtime, checkpoint_path, run_dir)
    skipped_incomplete = []

    for run_dir in sorted(glob.glob(os.path.join(runs_dir, "*"))):
        config_path = os.path.join(run_dir, "config.yml")
        ckpt_path = os.path.join(run_dir, f"checkpoint_{epoch:04d}.pth.tar")

        if not os.path.isfile(config_path):
            continue
        if not os.path.isfile(ckpt_path):
            skipped_incomplete.append(run_dir)
            continue

        with open(config_path, "r") as f:
            # CMID_project's utils.save_config_file dumps the raw argparse
            # Namespace via yaml.dump(args, ...) -- reading it back requires
            # the unsafe loader (it's a tagged Python object, not a plain
            # mapping).
            run_args = yaml.unsafe_load(f)
        dataset_name = getattr(run_args, "dataset_name", None)
        label = DATASET_NAME_MAP.get(dataset_name)
        if label is None:
            continue

        mtime = os.path.getmtime(ckpt_path)
        if label not in found or mtime > found[label][0]:
            found[label] = (mtime, ckpt_path, run_dir)

    if skipped_incomplete:
        print("Note: found run folder(s) without a completed "
              f"checkpoint_{epoch:04d}.pth.tar (still training or crashed), skipped:")
        for d in skipped_incomplete:
            print(f"    {d}")

    missing = [lbl for lbl in ("AID", "EuroSAT", "UC_Merced") if lbl not in found]
    if missing:
        raise FileNotFoundError(
            f"Missing completed CMID checkpoints for: {missing}\n"
            f"Found so far: { {k: v[2] for k, v in found.items()} }\n"
            f"Make sure all three CMID pretraining runs (aid, EuroSAT_RGB, "
            f"uc_merced) have finished all {epoch} epochs before running this script."
        )

    checkpoints = {label: path for label, (mtime, path, run_dir) in found.items()}
    print("Using CMID checkpoints:")
    for label, path in checkpoints.items():
        print(f"    {label:<10} -> {path}")
    return checkpoints


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


def run_one(checkpoint_path, data_dir):
    set_seed(SEED)
    train_loader, val_loader, test_loader, class_names = build_loaders(data_dir, SEED)

    model = get_model(checkpoint_path=checkpoint_path).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

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
            with torch.amp.autocast('cuda', enabled=USE_AMP):
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
                with torch.amp.autocast('cuda', enabled=USE_AMP):
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
            with torch.amp.autocast('cuda', enabled=USE_AMP):
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
    pretrain_checkpoints = discover_cmid_checkpoints(CMID_RUNS_DIR)
    base_dir = resolve_ms_dataset_base_dir()

    for split_name in SPLIT_NAMES:
        data_dir = os.path.join(base_dir, split_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Split folder not found: {data_dir}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    per_dataset_metrics = {}
    t0 = time.time()

    for name, checkpoint_path in pretrain_checkpoints.items():
        print(f"\n{'='*70}\nCMID BASELINE, pretrained on {name}, seed={SEED}\n{'='*70}")
        dataset_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        results_path = os.path.join(dataset_dir, "per_split_results.json")

        split_metrics = []
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                split_metrics = json.load(f)
        completed_splits = {m["split"] for m in split_metrics}

        for split_name in SPLIT_NAMES:
            if split_name in completed_splits:
                print(f"  [{name} | {split_name:>14}] already completed -- skipping (resume)")
                continue

            data_dir = os.path.join(base_dir, split_name)
            t_split = time.time()
            m, best_state = run_one(checkpoint_path, data_dir)
            m["split"] = split_name
            split_metrics.append(m)

            if SAVE_PER_RUN_CHECKPOINT:
                torch.save(best_state, os.path.join(dataset_dir, f"best_model_{split_name}.pth"))

            print(
                f"  [{name} | {split_name:>14}] acc={m['test_accuracy']*100:6.2f}%  "
                f"prec={m['test_precision']*100:6.2f}%  rec={m['test_recall']*100:6.2f}%  "
                f"f1={m['test_f1']*100:6.2f}%  "
                f"(best epoch {m['best_epoch']+1}/{m['epochs_trained']}, {time.time()-t_split:5.1f}s)"
            )

            with open(results_path, "w") as f:
                json.dump(split_metrics, f, indent=2)

        per_dataset_metrics[name] = split_metrics

    table_rows = []
    for name, split_metrics in per_dataset_metrics.items():
        accs = np.array([m["test_accuracy"] for m in split_metrics])
        precs = np.array([m["test_precision"] for m in split_metrics])
        recs = np.array([m["test_recall"] for m in split_metrics])
        f1s = np.array([m["test_f1"] for m in split_metrics])

        table_rows.append({
            "pretraining_dataset": name,
            "n_splits": len(split_metrics),
            "seed": SEED,
            "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
            "precision_mean": float(precs.mean()), "precision_std": float(precs.std()),
            "recall_mean": float(recs.mean()), "recall_std": float(recs.std()),
            "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
        })

    summary_path = os.path.join(RESULTS_DIR, "table1_cmid_5splits_summary.json")
    with open(summary_path, "w") as f:
        json.dump(table_rows, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTable I - CMID baseline (mean +/- std over {len(SPLIT_NAMES)} splits, seed={SEED}, %)\n{'='*70}")
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


if __name__ == "__main__":
    main()
