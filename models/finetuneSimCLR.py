import torch
import torch.nn as nn
from models.resnet_simclr import ResNetSimCLR
import config

def get_model(checkpoint_path=None):
    simclr_model = ResNetSimCLR(base_model='resnet50', out_dim=128)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state = checkpoint['state_dict']

        # Remove keys not needed for fine-tuning
        keys_to_remove = []

        for k in state.keys():
            if "projection_head" in k:
                keys_to_remove.append(k)
            if "classifier_head" in k:
                keys_to_remove.append(k)
            if "original_fc" in k:
                keys_to_remove.append(k)
            # Your model uses backbone.fc.*, so remove the FC of old model
            if "fc" in k and not k.startswith("backbone.fc"):
                keys_to_remove.append(k)

        # delete them
        for k in keys_to_remove:
            print("Removing:", k)
            del state[k]

        # Now load (missing keys are OK)
        simclr_model.load_state_dict(state, strict=False)

    encoder = simclr_model.backbone

    model = nn.Sequential(
        encoder,
        nn.Linear(128, config.NUM_CLASSES)
    )

    return model

