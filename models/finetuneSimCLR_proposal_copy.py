import torch
import torch.nn as nn
from models.resnet_simclr_proposal import ResNetSimCLR
import config

def get_model(checkpoint_path=None):
    # k_hat=32 matches the manuscript's stated K_hat and the actual
    # pre-training run (Section III-C, "Implementation Details"). Note this
    # value doesn't actually affect fine-tuning correctness either way: the
    # classifier_head is always stripped from the checkpoint below before
    # load_state_dict, since only the backbone (encoder) weights are reused
    # here -- classifier_head's shape is the only place k_hat matters
    # structurally, and it's discarded, not loaded.
    simclr_model = ResNetSimCLR(base_model='resnet50', out_dim=128, k_hat=32)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Delete incompatible classifier head params
        keys_to_delete = [k for k in checkpoint['state_dict'].keys()
                        if "classifier_head" in k]

        for k in keys_to_delete:
            print(f"Removing incompatible key: {k}")
            del checkpoint['state_dict'][k]

        # Delete original_fc if it exists
        for k in ["original_fc.weight", "original_fc.bias"]:
            if k in checkpoint['state_dict']:
                del checkpoint['state_dict'][k]

        simclr_model.load_state_dict(checkpoint['state_dict'], strict=False)


    encoder = simclr_model.backbone

    # Freeze encoder if needed
    if config.FREEZE_ENCODER:
        for param in encoder.parameters():
            param.requires_grad = False

    # Use the correct feature dimension (2048 for ResNet-50)
    model = nn.Sequential(
        encoder,
        nn.Linear(2048, config.NUM_CLASSES)  # Changed from 128 to 2048
    )

    return model