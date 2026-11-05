"""
R1 Major Comment 4 (part 2/2): "better yet, run evaluations using various
train-validation-test divisions."

Fine-tunes the proposed method across the 5 independently sampled MS_Dataset
splits (split_dataset1 .. split_dataset5 -- see
dataset/MS_Dataset/Build_multiple_splits.py), for EACH of the three
pretraining datasets (AID, EuroSAT, UC Merced), with the model-init seed held
FIXED (42) so the only thing varying run-to-run within a pretraining dataset
is the data split itself. This is the complement to
run_table1_proposal_5seeds.py, which holds the split fixed and varies the
seed. Together they rebuild all 3 rows of Table I's "Proposed method" block
under both robustness protocols the reviewer asked about.

15 fine-tuning runs total (3 pretraining datasets x 5 splits).

Same speed optimizations as the 5-seed script: mixed precision, multi-worker
pinned-memory DataLoaders, in-memory best-checkpoint tracking, minimal logging.

Usage:
    python run_table1_proposal_5splits.py
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

# --- OLD checkpoints (pre-fix: broken scheduler T_max, per-dataset k_hat,
#     no lambda warm-up) -- kept only for reference / diffing against the
#     new results, NOT used by this run. This is what produced the Table I
#     numbers where RSISC_SSL beat the Proposed method.
# PRETRAIN_CHECKPOINTS_OLD = {
#     "AID": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug20_02-40-55_DESKTOP-SU0KM9L_AID/checkpoint_0200.pth.tar"),
#     "EuroSAT": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug21_03-07-29_DESKTOP-SU0KM9L_EuroSAT/checkpoint_0200.pth.tar"),
#     "UC_Merced": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug25_01-20-02_DESKTOP-SU0KM9L_Uc_Merced/checkpoint_0200.pth.tar"),
# }

# --- NEW checkpoints (post-fix: T_max=epochs, k_hat=32 fixed, lambda
#     warm-up), currently pretraining on the lab PC (4090) under
#     ~/SarmadFolder/pkg/runs/<timestamp>_diamond/checkpoint_0200.pth.tar.
#     TODO once each run finishes: copy the three checkpoint_0200.pth.tar
#     files back from the lab PC into
#     D:/Scene classification/code/sc/entropy_simclr_full_project/runs/<same folder name>/
#     (or anywhere under GRSL_ROOT/code/sc -- just update the paths below to
#     match), then replace the placeholders "PENDING_..." with the real
#     folder names. Do NOT run this script until all three are filled in --
#     the startup check below will refuse to run on a "PENDING_..." path.
# FINAL, methodologically consistent set (task #35): all three pretraining
# datasets now use the SAME configuration -- batch-size=64, FP16 on,
# scheduler fix (T_max=epochs), K_hat=32 fixed, lambda warm-up. Batch=64
# was reverted from the earlier 256 because it was never one of the actual
# bug fixes (just an opportunistic increase enabled by the 4090's extra
# memory), and it was the prime remaining suspect for EuroSAT's regression
# under the fixed checkpoints. Verified all three checkpoints: 348MB each,
# zip structure valid, config.yml confirms batch_size=64/fp16_precision=true
# and the correct dataset_name for each.
PRETRAIN_CHECKPOINTS = {
    "AID": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug03_14-03-57_diamond/checkpoint_0200.pth.tar"),
    "EuroSAT": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug03_05-07-54_diamond/checkpoint_0200.pth.tar"),
    "UC_Merced": os.path.join(CODE_ROOT, "entropy_simclr_full_project/runs/Aug03_15-34-01_diamond/checkpoint_0200.pth.tar"),
}
SEED = 42                                        # fixed across all splits -- only the split varies
BASE_DIR = config.MS_DATASET_BASE_DIR
SPLIT_NAMES = config.MS_DATASET_SPLIT_NAMES      # split_dataset1 .. split_dataset5

# New folder, separate from every earlier attempt (_FIXED, _FIXED1, the
# EuroSAT-only diagnostic) -- this is the first run where all three
# datasets share one consistent pretraining configuration end to end.
RESULTS_DIR = "results_table1_proposal_5splits_BATCH64"
NUM_WORKERS = 0                                 # 0 avoids Windows multiprocessing/pagefile crashes; raise if your system handles it
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


def run_one(checkpoint_path, data_dir):
    set_seed(SEED)
    train_loader, val_loader, test_loader, class_names = build_loaders(data_dir, SEED)

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
    for name, path in PRETRAIN_CHECKPOINTS.items():
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint for '{name}' not found: {path}")

    for split_name in SPLIT_NAMES:
        data_dir = os.path.join(BASE_DIR, split_name)
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Split folder not found: {data_dir}\n"
                f"Run dataset/MS_Dataset/Build_multiple_splits.py first to generate all 5 splits."
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    per_dataset_metrics = {}  # pretrain-name -> list of per-split metric dicts
    t0 = time.time()

    for name, checkpoint_path in PRETRAIN_CHECKPOINTS.items():
        print(f"\n{'='*70}\nPROPOSED METHOD, pretrained on {name}, seed={SEED}\n{'='*70}")
        dataset_dir = os.path.join(RESULTS_DIR, name)
        os.makedirs(dataset_dir, exist_ok=True)

        results_path = os.path.join(dataset_dir, "per_split_results.json")

        # --- resume support: load whatever splits were already completed ---
        split_metrics = []
        if os.path.isfile(results_path):
            with open(results_path, "r") as f:
                split_metrics = json.load(f)
        completed_splits = {m["split"] for m in split_metrics}

        for split_name in SPLIT_NAMES:
            if split_name in completed_splits:
                print(f"  [{name} | {split_name:>14}] already completed -- skipping (resume)")
                continue

            data_dir = os.path.join(BASE_DIR, split_name)
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

            # save immediately after each split completes, not just at the end of the dataset,
            # so a crash/power loss only costs the split currently in progress
            with open(results_path, "w") as f:
                json.dump(split_metrics, f, indent=2)

        per_dataset_metrics[name] = split_metrics

    # --- aggregate mean +/- std per pretraining dataset (i.e. per Table I row) ---
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

    summary_path = os.path.join(RESULTS_DIR, "table1_proposed_5splits_summary.json")
    with open(summary_path, "w") as f:
        json.dump(table_rows, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nTable I - Proposed method (mean +/- std over {len(SPLIT_NAMES)} splits, seed={SEED}, %)\n{'='*70}")
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
    print(f"Per-split results under: {RESULTS_DIR}/<dataset>/per_split_results.json")


if __name__ == "__main__":
    main()
