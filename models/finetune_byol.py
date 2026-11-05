import torch
import torch.nn as nn
from torchvision.models import resnet50

import config


def get_model(checkpoint_path=None):
    """BYOL fine-tuning wrapper -- same convention as resnet_moco.py /
    resnet_simclr_proposal.py: build the plain downstream architecture (a
    ResNet-50 classifier), then load in only the relevant pretrained weights.

    BYOL_project saves the FULL model's state_dict (online_encoder +
    online_projector + online_predictor + target_encoder + target_projector,
    see BYOL_project/models/resnet_byol.py). For fine-tuning we keep ONLY
    online_encoder.* -- this is BYOL's own convention (Section 3.2 of the
    paper: downstream tasks always use the online encoder; the target network
    is just an EMA teacher with no gradient signal of its own and is a
    strictly "older"/laggier copy of the online weights, so there's no reason
    to fine-tune from it instead).
    """
    encoder = resnet50(weights=None)
    in_features = encoder.fc.in_features  # 2048, captured before replacing fc

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        prefix = "online_encoder."
        encoder_state = {
            k[len(prefix):]: v for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if not encoder_state:
            raise RuntimeError(
                f"No '{prefix}*' keys found in {checkpoint_path} -- wrong "
                f"checkpoint file, or BYOL's state_dict key names changed."
            )

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        # encoder.fc has no counterpart in online_encoder's saved state (BYOL
        # replaces it with nn.Identity() before pretraining -- see
        # BYOL_project/models/resnet_byol.py __init__), so "fc.weight"/"fc.bias"
        # missing is expected. Anything else missing, or any unexpected key,
        # means the architectures don't actually match and silently continuing
        # would just fine-tune from a random-init backbone.
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
