"""
Table II rebuild piece: cross-dataset generalization ablation. Models are
pre-trained (SSL) on EuroSAT and then fine-tuned + evaluated on UC_Merced
directly (dataset/uc_merced_split/{train,val,test}) -- NOT on the Mosul Scene
Dataset (MSD) splits that Table I uses, and NOT on EuroSAT itself. This tests
whether the learned representation transfers under domain shift to a
different downstream target, with no UC_Merced labels seen during
pretraining.

This is a genuinely different protocol from every run_baseline_*_5splits.py
script in this project (those all fine-tune+test on the SAME target, MSD,
varying only the pretraining source dataset). Table II's existing rows
(Supervised, SimCLR, MoCo, DCL, SimSiam, NCL, MECo, Proposed+ablations) were
computed as a SINGLE run (not the 5-split mean+-std protocol used for the
rebuilt Table I) -- this script matches that same single-run convention
(seed=42) so any new row it produces is comparable to the existing ones
rather than mixing single-run and mean+-std entries in the same table.

Currently supports:
    --method byol    (EuroSAT checkpoint already exists, see BYOL_project)
Barlow Twins is NOT supported yet -- it has no EuroSAT-pretrained checkpoint
(deliberately scoped to AID+UC_Merced only, same reasoning as SwAV before it).
Adding it here requires a new EuroSAT pretraining run for Barlow Twins first.

Setup (lab PC):
    export GRSL_ROOT=/home/erhan/SarmadFolder
    export BYOL_RUNS_DIR=/home/erhan/SarmadFolder/pkg/BYOL_project/runs   # if not at the default

Usage:
    python run_table2_crossdataset.py --method byol
"""

import argparse
import copy
import glob
import json
import os
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
from utils.metrics import compute_accuracy, compute_precision, compute_recall, compute_f1

SEED = 42
NUM_WORKERS = 0
USE_AMP = torch.cuda.is_available()

# --------------------------------------------------------------------------- #
# Per-method: (checkpoint discovery fn, finetune get_model fn, runs dir env var,
# default runs dir under GRSL_ROOT, run.py's -dataset-name -> our "EuroSAT" label)
# --------------------------------------------------------------------------- #

def _discover_checkpoint_for_label(runs_dir, want_label, dataset_name_map, epoch=200):
    """Shared discovery logic (same convention as every discover_<method>_
    checkpoints() function in this project's run_baseline_*_5splits.py
    scripts): scan runs_dir/*/config.yml for -dataset-name, matched against a
    completed checkpoint_<epoch>.pth.tar. Returns the newest completed run's
    checkpoint path for want_label, or raises with a clear message."""
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")

    found = None  # (mtime, path, run_dir)
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
            run_args = yaml.unsafe_load(f)
        dataset_name = getattr(run_args, "dataset_name", None)
        label = dataset_name_map.get(dataset_name)
        if label != want_label:
            continue

        mtime = os.path.getmtime(ckpt_path)
        if found is None or mtime > found[0]:
            found = (mtime, ckpt_path, run_dir)

    if skipped_incomplete:
        print("Note: found run folder(s) without a completed "
              f"checkpoint_{epoch:04d}.pth.tar, skipped:")
        for d in skipped_incomplete:
            print(f"    {d}")

    if found is None:
        raise FileNotFoundError(
            f"No completed {want_label}-pretrained checkpoint found in {runs_dir}"
        )

    print(f"Using checkpoint: {found[1]}")
    return found[1]


def get_method_config(method):
    if method == "byol":
        from models.finetune_byol import get_model
        runs_dir = os.environ.get(
            "BYOL_RUNS_DIR", os.path.join(config.GRSL_ROOT, "pkg/BYOL_project/runs")
        )
        dataset_name_map = {"aid": "AID", "EuroSAT_RGB": "EuroSAT", "uc_merced": "UC_Merced"}
        return get_model, runs_dir, dataset_name_map

    if method == "barlow":
        # Not usable yet -- see module docstring. Left in place so this
        # branch is a one-line change once a EuroSAT-pretrained Barlow Twins
        # checkpoint actually exists.
        from models.finetune_barlow import get_model
        runs_dir = os.environ.get(
            "BARLOW_RUNS_DIR", os.path.join(config.GRSL_ROOT, "pkg/BarlowTwins_project/runs")
        )
        dataset_name_map = {"aid": "AID", "EuroSAT_RGB": "EuroSAT", "uc_merced": "UC_Merced"}
        return get_model, runs_dir, dataset_name_map

    if method == "cmid":
        # Not usable yet either -- requires a EuroSAT-pretrained CMID
        # checkpoint from CMID_project/run.py -dataset-name EuroSAT_RGB first
        # (see CMID_project/README_LAB_PC.md). Same runs_dir/config.yml/
        # checkpoint_<epoch>.pth.tar discovery convention as every other
        # method above.
        from models.finetune_cmid import get_model
        runs_dir = os.environ.get(
            "CMID_RUNS_DIR", os.path.join(config.GRSL_ROOT, "pkg/CMID_project/runs")
        )
        dataset_name_map = {"aid": "AID", "EuroSAT_RGB": "EuroSAT", "uc_merced": "UC_Merced"}
        return get_model, runs_dir, dataset_name_map

    if method == "proposed":
        # Corrected Proposed-method pretraining (batch=256, k_hat=32 fixed,
        # lambda warm-up+cosine schedule fixed) -- the same lab-machine run
        # that produced the AID checkpoint used for Table I's rebuilt
        # numbers (confirmed by the user: batch=256, k_hat=32). EuroSAT was
        # pretrained in that same run, at
        # pkg/runs1/Aug02_18-39-00_diamond/checkpoint_0200.pth.tar -- used
        # here so Table II's Proposed row is finally consistent with
        # Table I's corrected recipe (the OLD Table II Proposed row predates
        # this fix and was computed by an earlier, separate script, not this
        # one).
        from models.finetuneSimCLR_proposal_copy import get_model
        runs_dir = os.environ.get(
            "PROPOSED_RUNS_DIR", os.path.join(config.GRSL_ROOT, "pkg/runs1")
        )
        dataset_name_map = {"aid": "AID", "EuroSAT_RGB": "EuroSAT", "uc_merced": "UC_Merced"}
        return get_model, runs_dir, dataset_name_map

    raise ValueError(f"Unknown method: {method}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_uc_merced_split_dir():
    """dataset/uc_merced_split/{train,val,test} -- the LABELED, held-out-split
    UC_Merced folder used for downstream fine-tuning/testing. Distinct from
    dataset/uc_merced/ (flat, unlabeled-for-pretraining folder used by every
    SSL project's -dataset-name uc_merced)."""
    candidate = os.path.join(config.GRSL_ROOT, "dataset/uc_merced_split")
    if all(os.path.isdir(os.path.join(candidate, s)) for s in ("train", "val", "test")):
        return candidate

    for dirpath, dirnames, _ in os.walk(config.GRSL_ROOT):
        if os.path.basename(dirpath).lower() == "uc_merced_split" and all(
            s in dirnames for s in ("train", "val", "test")
        ):
            return dirpath

    raise FileNotFoundError(
        f"Could not find uc_merced_split/{{train,val,test}} under "
        f"GRSL_ROOT={config.GRSL_ROOT}"
    )


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


def run_one(get_model, checkpoint_path, data_dir):
    set_seed(SEED)
    train_loader, val_loader, test_loader, class_names = build_loaders(data_dir, SEED)
    print(f"    classes ({len(class_names)}): {class_names}")

    # config.NUM_CLASSES defaults to 9 (the Mosul Scene Dataset used by every
    # Table I run_baseline_*_5splits.py script). Table II's target is
    # UC_Merced instead (21 classes) -- override before building the model's
    # final layer so get_model()'s nn.Linear(in_features, config.NUM_CLASSES)
    # actually matches the real label range, instead of silently truncating
    # to 9 outputs and crashing on any label >= 9.
    config.NUM_CLASSES = len(class_names)

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
    parser = argparse.ArgumentParser(description="Table II cross-dataset ablation (EuroSAT -> UC_Merced)")
    parser.add_argument("--method", required=True, choices=["byol", "barlow", "cmid", "proposed"])
    args = parser.parse_args()

    get_model, runs_dir, dataset_name_map = get_method_config(args.method)
    checkpoint_path = _discover_checkpoint_for_label(runs_dir, "EuroSAT", dataset_name_map)
    data_dir = resolve_uc_merced_split_dir()

    print(f"\n{'='*70}\nTABLE II cross-dataset: {args.method} pretrained on EuroSAT, "
          f"evaluated on UC_Merced, seed={SEED}\n{'='*70}")

    results_dir = f"results_table2_{args.method}_crossdataset"
    os.makedirs(results_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    t0 = time.time()
    metrics, best_state = run_one(get_model, checkpoint_path, data_dir)
    torch.save(best_state, os.path.join(results_dir, "best_model.pth"))

    result_row = {
        "method": args.method,
        "pretraining_dataset": "EuroSAT",
        "eval_dataset": "UC_Merced",
        "seed": SEED,
        "test_accuracy": metrics["test_accuracy"],
        "test_precision": metrics["test_precision"],
        "test_recall": metrics["test_recall"],
        "test_f1": metrics["test_f1"],
        "best_epoch": metrics["best_epoch"],
        "epochs_trained": metrics["epochs_trained"],
    }
    summary_path = os.path.join(results_dir, f"table2_{args.method}_crossdataset_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result_row, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTable II row -- {args.method}, EuroSAT -> UC_Merced (%)\n{'='*70}")
    print(
        f"acc={metrics['test_accuracy']*100:.2f}  prec={metrics['test_precision']*100:.2f}  "
        f"rec={metrics['test_recall']*100:.2f}  f1={metrics['test_f1']*100:.2f}  "
        f"(best epoch {metrics['best_epoch']+1}/{metrics['epochs_trained']}, {elapsed:.1f}s)"
    )
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
