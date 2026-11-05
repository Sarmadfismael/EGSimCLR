import torch
import torch.nn as nn
from torchvision.models import resnet50

import config


def get_model(checkpoint_path=None):
    """SwAV fine-tuning wrapper -- same convention as finetune_byol.py /
    resnet_moco.py: build the plain downstream architecture (a ResNet-50
    classifier), then load in only the encoder weights.

    SwAV_project saves the FULL model's state_dict (encoder + projector +
    prototypes, see SwAV_project/models/resnet_swav.py). For fine-tuning we
    keep ONLY encoder.* -- the projector and prototypes exist purely to
    define the pretraining objective (mapping to cluster prototypes); the
    encoder is what carries the transferable visual representation. This
    matches SwAV's own downstream evaluation convention (Section 4.1 of the
    paper: linear classification and finetuning are both done on the encoder
    output, never through the projector/prototypes).
    """
    encoder = resnet50(weights=None)
    in_features = encoder.fc.in_features  # 2048, captured before replacing fc

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        prefix = "encoder."
        encoder_state = {
            k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError(
                f"No '{prefix}*' keys found in {checkpoint_path} -- wrong "
                f"checkpoint file, or SwAV's state_dict key names changed."
            )

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        # encoder.fc has no counterpart in the saved state (SwAV replaces it
        # with nn.Identity() before pretraining -- see
        # SwAV_project/models/resnet_swav.py __init__), so "fc.weight"/
        # "fc.bias" missing is expected. Anything else missing, or any
        # unexpected key, means the architectures don't actually match.
        bad_missing = [k for k in missing if not k.startswith("fc.")]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/architecture mismatch loading {checkpoint_path}\n"
                f"  unexpected missing keys: {bad_missing}\n"
                f"  unexpected extra keys:   {list(unexpected)}"
            )

    if config.FREEZE_ENCODER:
        for p in encoder.parameters():
            p.requires_grad = False

    encoder.fc = nn.Linear(in_features, config.NUM_CLASSES)
    return encoder
