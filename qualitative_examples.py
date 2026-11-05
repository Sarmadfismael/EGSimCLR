"""
R1 Major Comment 8 (qualitative examples): find real MSD test images where
Proposed corrects a SimCLR misclassification (both AID pre-training),
pooled across all 5 splits. Given the strict 5-page GRSL limit, this output
goes into the REBUTTAL TEXT as described examples rather than a manuscript
figure (see Response_to_Reviewers.tex, Major Comment 8).

Also tracks the reverse case (SimCLR right, Proposed wrong) and cases both
get wrong, for an honest picture rather than cherry-picked wins only.

Usage (from inside GRSL_Project/, same environment as the other 5-split
scripts):
    python qualitative_examples.py
"""
import json
import os

import torch
from torchvision import transforms, datasets

from models.finetuneSimCLR import get_model as get_simclr_model
from models.finetuneSimCLR_proposal_copy import get_model as get_proposed_model
import config

SIMCLR_DIR = "results_table1_simclr_5splits/AID"
PROPOSED_DIR = "results_table1_proposal_5splits_FIXED/AID"


def build_test_set(data_dir):
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)


@torch.no_grad()
def predict_all(model, test_set):
    model.eval()
    # shuffle=False -- prediction order must match test_set.samples order
    # exactly so we can map each prediction back to its image path.
    loader = torch.utils.data.DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    preds = []
    for inputs, _ in loader:
        inputs = inputs.to(config.DEVICE)
        outputs = model(inputs)
        preds.extend(outputs.argmax(dim=1).cpu().tolist())
    return preds


def main():
    corrected = []               # SimCLR wrong, Proposed right
    still_missed = []            # both wrong
    proposed_introduced_error = []  # SimCLR right, Proposed wrong (honesty check)

    for split_name in config.MS_DATASET_SPLIT_NAMES:
        data_dir = os.path.join(config.MS_DATASET_BASE_DIR, split_name)
        test_set = build_test_set(data_dir)
        class_names = test_set.classes

        simclr_model = get_simclr_model(checkpoint_path=None).to(config.DEVICE)
        simclr_model.load_state_dict(torch.load(
            os.path.join(SIMCLR_DIR, f"best_model_{split_name}.pth"), map_location=config.DEVICE))

        proposed_model = get_proposed_model(checkpoint_path=None).to(config.DEVICE)
        proposed_model.load_state_dict(torch.load(
            os.path.join(PROPOSED_DIR, f"best_model_{split_name}.pth"), map_location=config.DEVICE))

        simclr_preds = predict_all(simclr_model, test_set)
        proposed_preds = predict_all(proposed_model, test_set)

        split_corrected = split_missed = split_regressed = 0
        for idx, (path, true_label) in enumerate(test_set.samples):
            s_pred, p_pred = simclr_preds[idx], proposed_preds[idx]
            record = {
                "split": split_name,
                "image_path": path,
                "true_label": class_names[true_label],
                "simclr_pred": class_names[s_pred],
                "proposed_pred": class_names[p_pred],
            }
            if s_pred != true_label and p_pred == true_label:
                corrected.append(record)
                split_corrected += 1
            elif s_pred != true_label and p_pred != true_label:
                still_missed.append(record)
                split_missed += 1
            elif s_pred == true_label and p_pred != true_label:
                proposed_introduced_error.append(record)
                split_regressed += 1

        print(f"{split_name}: corrected={split_corrected}  still_missed={split_missed}  "
              f"proposed_introduced_error={split_regressed}")

    print(f"\nTotal corrected (SimCLR wrong, Proposed right): {len(corrected)}")
    print(f"Total still missed (both wrong): {len(still_missed)}")
    print(f"Total proposed introduced error (SimCLR right, Proposed wrong): {len(proposed_introduced_error)}")

    with open("qualitative_examples.json", "w") as f:
        json.dump({
            "corrected": corrected,
            "still_missed": still_missed,
            "proposed_introduced_error": proposed_introduced_error,
        }, f, indent=2)
    print("\nSaved: qualitative_examples.json")

    print("\nSample corrected cases (up to 8):")
    for r in corrected[:8]:
        print(f"  [{r['split']}] {os.path.basename(r['image_path'])}: true={r['true_label']}, "
              f"SimCLR={r['simclr_pred']} (wrong), Proposed={r['proposed_pred']} (correct)")

    if still_missed:
        print("\nSample still-missed cases (up to 5):")
        for r in still_missed[:5]:
            print(f"  [{r['split']}] {os.path.basename(r['image_path'])}: true={r['true_label']}, "
                  f"SimCLR={r['simclr_pred']}, Proposed={r['proposed_pred']} (both wrong)")

    if proposed_introduced_error:
        print("\nProposed-introduced-error cases (Proposed regressed vs SimCLR):")
        for r in proposed_introduced_error:
            print(f"  [{r['split']}] {os.path.basename(r['image_path'])}: true={r['true_label']}, "
                  f"SimCLR={r['simclr_pred']} (correct), Proposed={r['proposed_pred']} (wrong)")


if __name__ == "__main__":
    main()
