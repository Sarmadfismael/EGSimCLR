import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets.loader import get_loaders
from models.resnet50 import get_model
#from models.finetuneSimCLR import get_model
#from models.finetuneSimCLR_proposal_copy import get_model
#from models.resnet_moco import get_model

from utils.metrics import (
    compute_accuracy,
    compute_precision,
    compute_recall,
    compute_f1,
    compute_confusion_matrix
)
import config
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm


def setup_logger(output_dir):
    """
    Build a logger that writes to <output_dir>/training.log. A fresh logger
    (not the root logger) is used so that running train() back-to-back for
    several dataset splits doesn't leak handlers/log lines between splits.
    """
    logger = logging.getLogger(f"train.{os.path.abspath(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'), mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# Save confusion matrix image
def plot_confusion_matrix(cm, class_names, output_dir, epoch=None, normalize=True,
                           title="Confusion Matrix", filename=None):
    """
    Plot and save a confusion matrix heatmap with a color bar indicating value range.

    Args:
        cm (np.array): Confusion matrix.
        class_names (list): List of class names.
        output_dir (str): Directory to save the plot into.
        epoch (int, optional): Epoch number for title and filename.
        normalize (bool, optional): Whether to normalize the confusion matrix.
        title (str, optional): Title of the plot.
        filename (str, optional): Filename to save the plot.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Row-wise normalization
        fmt = ".1f"
        cmap = "Blues"
    else:
        fmt = "d"
        cmap = "Reds"

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        cbar=True,  # Enable color bar (sidebar)
        cbar_kws={"shrink": 0.8},  # Slightly shrink color bar for better fit
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

    if epoch is not None:
        ax.set_title(f"{title} (Epoch {epoch+1})", fontsize=16, pad=12)
        if filename is None:
            filename = f"conf_matrix_epoch_{epoch+1}.pdf"
    else:
        ax.set_title(title, fontsize=16, pad=12)
        if filename is None:
            filename = "conf_matrix.pdf"

    plt.tight_layout()
   # plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# Save training/validation curves
def plot_training_curves(train_accs, val_accs, train_losses, val_losses, output_dir, best_epoch=None):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    if best_epoch is not None:
        plt.axvline(x=best_epoch + 1, color='r', linestyle='--', label='Early Stopping Point')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    if best_epoch is not None:
        plt.axvline(x=best_epoch + 1, color='r', linestyle='--', label='Early Stopping Point')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def train(data_dir=None, output_dir="."):
    """
    Train the ResNet-50 baseline on the dataset split at `data_dir`
    (an ImageFolder root with train/val/test subfolders).

    All artifacts (training.log, best_model.pth, training_curves.pdf,
    confusion matrices) are written into `output_dir`, so multiple splits
    can be trained one after another without overwriting each other's results.

    Returns:
        (best_model_path, class_names)
    """
    data_dir = data_dir or config.DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)
    best_model_path = os.path.join(output_dir, "best_model.pth")

    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")

    train_loader, val_loader, _, class_names = get_loaders(data_dir)
    model = get_model().to(config.DEVICE)
    #model = get_model(checkpoint_path='D:/Scene classification/code/sc/entropy_simclr_full_project/runs/Aug21_03-07-29_DESKTOP-SU0KM9L_EuroSAT/checkpoint_0200.pth.tar').to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    best_epoch = None

    best_val_acc = 0.0
    epochs_no_improve = 0

    train_accs, val_accs = [], []
    train_losses, val_losses = [], []

    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            train_loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(avg_loss)
        train_accs.append(train_acc)
        logger.info(f"Epoch [{epoch+1}/{config.EPOCHS}] - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc*100:.2f}%")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_running_loss = 0.0

        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Val]", leave=False)
            for val_inputs, val_targets in val_loop:
                val_inputs = val_inputs.to(config.DEVICE)
                val_targets = val_targets.to(config.DEVICE)
                outputs = model(val_inputs)
                loss = criterion(outputs, val_targets)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(val_targets.cpu().numpy())

        val_acc = compute_accuracy(val_labels, val_preds)
        val_precision = compute_precision(val_labels, val_preds)
        val_recall = compute_recall(val_labels, val_preds)
        val_f1 = compute_f1(val_labels, val_preds)
        val_loss = val_running_loss / len(val_loader)

        val_accs.append(val_acc)
        val_losses.append(val_loss)

        logger.info(f"Validation - Acc: {val_acc*100:.2f}%, Precision: {val_precision*100:.2f}%, Recall: {val_recall*100:.2f}%, F1: {val_f1*100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with accuracy: {val_acc*100:.2f}%")
            best_epoch = epoch
            epochs_no_improve = 0

            cm = compute_confusion_matrix(val_labels, val_preds)
            plot_confusion_matrix(
                cm,
                class_names,
                output_dir,
                epoch=epoch,
                normalize=True,
                title="Normalized Confusion Matrix",
            )
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve == config.patience:
            logger.info("Early stopping triggered!")
            break

    plot_training_curves(train_accs, val_accs, train_losses, val_losses, output_dir, best_epoch)

    # Detach handlers so the next split's logger starts clean (file handle release)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    return best_model_path, class_names


if __name__ == "__main__":
    train(config.DATA_DIR, ".")
