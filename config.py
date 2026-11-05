
import os
import torch

# Single override point for moving machines (e.g. Windows dev PC -> Ubuntu lab
# PC). Set the env var GRSL_ROOT to the new machine's equivalent of
# "D:/Scene classification" (the parent of both "dataset/" and "code/sc/"),
# preserving the same relative folder layout underneath it. Everything else
# in this file, and every run_*_5splits.py script that reads
# config.GRSL_ROOT / config.MS_DATASET_BASE_DIR, derives from this one value.
GRSL_ROOT = os.environ.get("GRSL_ROOT", "D:/Scene classification")

DATA_DIR = os.path.join(GRSL_ROOT, "dataset/MS_Dataset/split_dataset1")
NUM_CLASSES = 9
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4
IMAGE_SIZE = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
patience = 20  # Early stopping patience
FREEZE_ENCODER = False

# --- 5-split robustness evaluation (MS_Dataset) ---
# These are the 5 independently sampled train/val/test splits produced by
# dataset/MS_Dataset/Build_multiple_splits.py (different random_state per split),
# used to report mean +/- std results instead of a single train/test split.
MS_DATASET_BASE_DIR = os.path.join(GRSL_ROOT, "dataset/MS_Dataset")
MS_DATASET_SPLIT_NAMES = [
    "split_dataset1",  # seed 42
    "split_dataset2",  # seed 123
    "split_dataset3",  # seed 456
    "split_dataset4",  # seed 789
    "split_dataset5",  # seed 1011
]
RESULTS_ROOT = "results_MS_5splits"
