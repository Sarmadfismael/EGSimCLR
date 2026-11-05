import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from models.resnet50 import get_model
#from models.finetuneSimCLR import get_model
#from models.finetuneSimCLR_proposal_copy import get_model
#from models.resnet_moco import get_model
from datasets.loader import get_loaders
from utils.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix,
)
import config


def plot_confusion_matrix(cm, class_names, normalize=True, title="Confusion Matrix", filename="confusion_matrix_test.pdf"):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
        annot = np.array([[f"{cell:.1f}%" for cell in row] for row in cm])
        fmt = ""
        cmap = "Blues"
    else:
        annot = True
        fmt = "d"
        cmap = "Reds"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt=fmt if normalize else "d",
        cmap=cmap,
        cbar=True,  # Enable sidebar
        cbar_kws={"shrink": 0.8, "label": "Percentage (%)" if normalize else "Count"},
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 12, "weight": "bold"}
    )

    ax.set_xlabel("Predicted Label", fontsize=14, labelpad=10)
    ax.set_ylabel("True Label", fontsize=14, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
    ax.set_title(title, fontsize=16, pad=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(data_dir=None, output_dir=".", model_path=None):
    """
    Evaluate the trained model on the test split of `data_dir`.

    Args:
        data_dir: dataset split root (train/val/test subfolders). Defaults to config.DATA_DIR.
        output_dir: where to save the test confusion matrix PDF.
        model_path: checkpoint to load. Defaults to <output_dir>/best_model.pth.

    Returns:
        dict with keys: accuracy, precision, recall, f1, confusion_matrix
    """
    data_dir = data_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    model_path = model_path or os.path.join(output_dir, "best_model.pth")

    _, _, test_loader, class_names = get_loaders(data_dir)

    model = get_model().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            labels.extend(targets.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    acc = compute_accuracy(labels, preds)
    precision = compute_precision(labels, preds)
    recall = compute_recall(labels, preds)
    f1 = compute_f1(labels, preds)
    cm = compute_confusion_matrix(labels, preds)

    # Per-class accuracy = diagonal of the row-normalized confusion matrix
    # (each row is "given true class i, fraction predicted as each class" --
    # the diagonal entry is exactly per-class recall/accuracy). R1 Major
    # Comment 8 asks for this explicitly, alongside the confusion matrix
    # itself (which already implies it visually, annotated as %).
    cm_float = cm.astype("float")
    row_sums = cm_float.sum(axis=1)
    per_class_acc = np.divide(
        cm_float.diagonal(), row_sums,
        out=np.zeros_like(cm_float.diagonal()), where=row_sums != 0,
    )

    print(f"Test Accuracy  : {acc * 100:.2f}%")
    print(f"Test Precision : {precision * 100:.2f}%")
    print(f"Test Recall    : {recall * 100:.2f}%")
    print(f"Test F1 Score  : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nPer-Class Accuracy:")
    for name, pca in sorted(zip(class_names, per_class_acc), key=lambda x: -x[1]):
        print(f"  {name:<20s} {pca * 100:6.2f}%")

    plot_confusion_matrix(
        cm,
        class_names,
        normalize=True,
        title="Normalized Confusion Matrix (Test Set)",
        filename=os.path.join(output_dir, "conf_matrix_test.pdf"),
    )

    per_class_path = os.path.join(output_dir, "per_class_accuracy.json")
    with open(per_class_path, "w") as f:
        json.dump(
            {name: float(pca) for name, pca in zip(class_names, per_class_acc)},
            f, indent=2,
        )
    print(f"\nSaved per-class accuracy: {per_class_path}")
    print(f"Saved confusion matrix PDF: {os.path.join(output_dir, 'conf_matrix_test.pdf')}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "per_class_accuracy": dict(zip(class_names, per_class_acc.tolist())),
    }


if __name__ == "__main__":
    evaluate(config.DATA_DIR, ".")
