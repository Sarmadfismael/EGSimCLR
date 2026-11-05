import torch
import torch.nn as nn
from torchvision.models import resnet50

import config


def get_model(checkpoint_path=None):
    """CMID fine-tuning wrapper -- same convention as finetune_byol.py /
    finetune_barlow.py: build the plain downstream ResNet-50 classifier, then
    load in only the student encoder's weights.

    Paper (end of Section III-A): "After pretraining using CMID with a large
    amount of unlabeled data, only the student is used for downstream tasks
    to evaluate the model performance." The teacher encoder, MIM head, global/
    local projectors, and prototypes are all pretraining-only scaffolding and
    are discarded here -- this mirrors the official repo's extract.py, which
    strips everything except online_encoder.model.*; our
    CMID_project/models/resnet_cmid.py's equivalent prefix is
    "student_encoder.".

    CMID_project's ResNetFeatureExtractor splits the backbone into
    stem (=conv1+bn1+relu+maxpool) / layer1-4 rather than torchvision's flat
    resnet50 attribute names, so stem.0./stem.1. are remapped back to
    conv1./bn1. below (relu/maxpool have no learnable params, so they need no
    entry). layer1-4 already match torchvision's naming directly.
    """
    encoder = resnet50(weights=None)
    in_features = encoder.fc.in_features  # 2048

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint)

        prefix = "student_encoder."
        remap = {"stem.0.": "conv1.", "stem.1.": "bn1."}

        encoder_state = {}
        for k, v in state_dict.items():
            if not k.startswith(prefix):
                continue
            key = k[len(prefix):]
            for old, new in remap.items():
                if key.startswith(old):
                    key = new + key[len(old):]
                    break
            encoder_state[key] = v

        if not encoder_state:
            raise RuntimeError(
                f"No '{prefix}*' keys found in {checkpoint_path} -- wrong "
                f"checkpoint file, or CMID_project's state_dict key names changed."
            )

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        # encoder.fc has no counterpart in student_encoder's saved state (the
        # feature extractor has no fc layer at all -- see resnet_cmid.py), so
        # "fc.weight"/"fc.bias" missing is expected. Anything else missing,
        # or any unexpected key, means the architectures don't actually match.
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
