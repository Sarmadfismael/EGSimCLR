import torch
from models.builder import MoCo
from torchvision.models import resnet50
import torch.nn as nn
import config

def get_model(checkpoint_path=None):
    # Create MoCo model (architecture must match your pretraining)
    moco_model = MoCo(base_encoder=resnet50, dim=128, K=65536, m=0.999, T=0.07, mlp=True)

    # Load checkpoint if provided
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        moco_model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Use encoder_q for downstream tasks
    encoder = moco_model.encoder_q

    # Freeze encoder if needed
    if config.FREEZE_ENCODER:
        for param in encoder.parameters():
            param.requires_grad = False

    # --- FIX: handle Sequential or Linear fc ---
    if isinstance(encoder.fc, nn.Sequential):
        in_features = encoder.fc[0].in_features
    else:
        in_features = encoder.fc.in_features

    # Replace projection head with classification layer
    encoder.fc = nn.Linear(in_features, config.NUM_CLASSES)

    return encoder
