"""
R1 Major Comment 8: aggregated confusion matrix + per-class accuracy for MSD.

Rather than picking one representative split, this pools raw (count)
confusion matrices across all 5 splits' held-out test sets before
normalizing -- each split has its own fine-tuned checkpoint
(best_model_split_dataset{1-5}.pth) evaluated on its own test fold, so
summing element-wise before normalizing gives ~75 test images/class instead
of ~15 (R1 Major Comment 4 already flagged the 15-images/class test set as
small; this reuses the same 5-split infrastructure to make the confusion
matrix more statistically stable too, not just the headline accuracy).

Usage (from inside GRSL_Project/, same environment used for
run_table1_proposal_5splits.py):
    python evaluate_5split_confusion.py \
        --checkpoints-dir results_table1_proposal_5splits_FIXED/AID \
        --output-dir results_r1_confusion_matrix
"""
import argparse
import json
import os

import numpy as np
import torch

# NOT models.resnet50 -- run_table1_proposal_5splits.py (which actually
# produced these checkpoints) builds the Proposed method's model via
# finetuneSimCLR_proposal_copy.get_model(), which returns
# nn.Sequential(ResNetSimCLR(...).backbone, nn.Linear(2048, NUM_CLASSES)).
# That's the architecture best_model_split_dataset*.pth's keys ("0.*" for
# the backbone, "1.weight"/"1.bias" for the head) actually match.
from models.finetuneSimCLR_proposal_copy import get_model
from datasets.loader import get_loaders
from utils.metrics import compute_confusion_matrix
from test import plot_confusion_matrix
import config


def evaluate_one_split(checkpoint_path, data_dir):
    _, _, test_loader, class_names = get_loaders(data_dir)

    # checkpoint_path=None here builds the architecture only (random init,
    # not the SSL-pretraining checkpoint) -- the actual fine-tuned weights
    # (encoder + trained classifier head) are loaded via load_state_dict
    # below, straight from the already-fine-tuned best_model_split_dataset*.pth.
    model = get_model(checkpoint_path=None).to(config.DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.numpy())

    cm = compute_confusion_matrix(np.array(labels), np.array(preds))
    return cm, class_names


def main():
    parser = argparse.ArgumentParser(description="Aggregate MSD confusion matrix across all 5 splits")
    parser.add_argument("--checkpoints-dir", required=True,
                         help="Folder containing best_model_split_dataset{1-5}.pth (e.g. results_table1_proposal_5splits_FIXED/AID)")
    parser.add_argument("--output-dir", default="results_r1_confusion_matrix")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_cm = None
    reference_class_names = None

    for split_name in config.MS_DATASET_SPLIT_NAMES:
        checkpoint_path = os.path.join(args.checkpoints_dir, f"best_model_{split_name}.pth")
        data_dir = os.path.join(config.MS_DATASET_BASE_DIR, split_name)

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        print(f"Evaluating {split_name} -- checkpoint: {checkpoint_path}")
        cm, class_names = evaluate_one_split(checkpoint_path, data_dir)

        if reference_class_names is None:
            reference_class_names = class_names
        elif class_names != reference_class_names:
            raise RuntimeError(
                f"Class name/order mismatch on {split_name}: {class_names} != {reference_class_names} "
                f"-- ImageFolder's alphabetical class ordering should be identical across splits since "
                f"they're the same 9 MSD categories; something is off if this fires."
            )

        total_cm = cm if total_cm is None else total_cm + cm

    print(f"\nPooled {len(config.MS_DATASET_SPLIT_NAMES)} splits, "
          f"{total_cm.sum()} total test predictions "
          f"({total_cm.sum() // len(reference_class_names)} avg per class).")

    # Per-class accuracy = diagonal of the row-normalized POOLED matrix.
    cm_float = total_cm.astype("float")
    row_sums = cm_float.sum(axis=1)
    per_class_acc = np.divide(
        cm_float.diagonal(), row_sums,
        out=np.zeros_like(cm_float.diagonal()), where=row_sums != 0,
    )
    overall_acc = cm_float.diagonal().sum() / cm_float.sum()

    print(f"\nPooled overall accuracy: {overall_acc * 100:.2f}%")
    print("\nPer-Class Accuracy (pooled across 5 splits):")
    for name, pca in sorted(zip(reference_class_names, per_class_acc), key=lambda x: -x[1]):
        print(f"  {name:<20s} {pca * 100:6.2f}%")

    plot_confusion_matrix(
        total_cm,
        reference_class_names,
        normalize=True,
        title="Normalized Confusion Matrix (MSD, pooled across 5 splits)",
        filename=os.path.join(args.output_dir, "conf_matrix_msd_5split_pooled.pdf"),
    )

    with open(os.path.join(args.output_dir, "confusion_matrix_5split_pooled.json"), "w") as f:
        json.dump({
            "class_names": reference_class_names,
            "pooled_confusion_matrix": total_cm.tolist(),
            "per_class_accuracy": {name: float(pca) for name, pca in zip(reference_class_names, per_class_acc)},
            "overall_accuracy": float(overall_acc),
            "n_splits_pooled": len(config.MS_DATASET_SPLIT_NAMES),
            "total_test_predictions": int(total_cm.sum()),
        }, f, indent=2)

    print(f"\nSaved: {os.path.join(args.output_dir, 'conf_matrix_msd_5split_pooled.pdf')}")
    print(f"Saved: {os.path.join(args.output_dir, 'confusion_matrix_5split_pooled.json')}")


if __name__ == "__main__":
    main()
