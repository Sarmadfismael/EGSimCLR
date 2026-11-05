"""
Run the ResNet-50 baseline (train.py / test.py) across all 5 MS_Dataset splits
(split_dataset1, split_dataset2, split_dataset3, split_dataset4, split_dataset5)
and report mean +/- std accuracy/precision/recall/F1 across the 5 runs.

This addresses the reviewer concern that a single train/val/test split can
give an overly optimistic or unlucky estimate of performance on a small,
custom dataset. Each split is an independently sampled 70/15/15 partition
(see dataset/MS_Dataset/Build_multiple_splits.py for how they were built).

Usage:
    python run_five_splits.py
"""

import os
import csv
import json
import numpy as np

import config
from train import train
from test import evaluate


def main():
    base_dir = config.MS_DATASET_BASE_DIR
    split_names = config.MS_DATASET_SPLIT_NAMES
    results_root = config.RESULTS_ROOT
    os.makedirs(results_root, exist_ok=True)

    all_metrics = []

    for split_name in split_names:
        data_dir = os.path.join(base_dir, split_name)
        output_dir = os.path.join(results_root, split_name)

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Split folder not found: {data_dir}\n"
                f"Run dataset/MS_Dataset/Build_multiple_splits.py first to generate all 5 splits."
            )

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*70}\nTRAINING on {split_name}  ({data_dir})\n{'='*70}")
        best_model_path, class_names = train(data_dir=data_dir, output_dir=output_dir)

        print(f"\n{'-'*70}\nEVALUATING on {split_name}\n{'-'*70}")
        metrics = evaluate(data_dir=data_dir, output_dir=output_dir, model_path=best_model_path)

        row = {
            "split": split_name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }
        all_metrics.append(row)

        print(f"\n{split_name} -> Acc: {row['accuracy']*100:.2f}%  "
              f"P: {row['precision']*100:.2f}%  R: {row['recall']*100:.2f}%  F1: {row['f1']*100:.2f}%")

    # --- Per-split CSV ---
    csv_path = os.path.join(results_root, "per_split_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "accuracy", "precision", "recall", "f1"])
        writer.writeheader()
        for row in all_metrics:
            writer.writerow(row)

    # --- Aggregate mean +/- std ---
    accs = np.array([m["accuracy"] for m in all_metrics])
    precs = np.array([m["precision"] for m in all_metrics])
    recs = np.array([m["recall"] for m in all_metrics])
    f1s = np.array([m["f1"] for m in all_metrics])

    summary = {
        "n_splits": len(all_metrics),
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "precision_mean": float(precs.mean()), "precision_std": float(precs.std()),
        "recall_mean": float(recs.mean()), "recall_std": float(recs.std()),
        "f1_mean": float(f1s.mean()), "f1_std": float(f1s.std()),
    }

    summary_json_path = os.path.join(results_root, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    summary_txt_path = os.path.join(results_root, "summary.txt")
    lines = [
        f"Results across {summary['n_splits']} dataset splits (mean +/- std, %)",
        f"Accuracy  : {summary['accuracy_mean']*100:.2f} +/- {summary['accuracy_std']*100:.2f}",
        f"Precision : {summary['precision_mean']*100:.2f} +/- {summary['precision_std']*100:.2f}",
        f"Recall    : {summary['recall_mean']*100:.2f} +/- {summary['recall_std']*100:.2f}",
        f"F1        : {summary['f1_mean']*100:.2f} +/- {summary['f1_std']*100:.2f}",
    ]
    with open(summary_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nPer-split results: {csv_path}")
    print(f"Summary (txt)    : {summary_txt_path}")
    print(f"Summary (json)   : {summary_json_path}")


if __name__ == "__main__":
    main()
