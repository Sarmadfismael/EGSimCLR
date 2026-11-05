import torch
import torch.nn as nn
from models.resnet_simclr_proposal import ResNetSimCLR
import config

def get_model(checkpoint_path=None):
    # Use the same k_hat value that was used during pre-training
    simclr_model = ResNetSimCLR(base_model='resnet50', out_dim=128, k_hat=25)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle missing 'original_fc' weights
        if 'original_fc.weight' in checkpoint['state_dict']:
            # Remove original_fc weights from checkpoint if they exist
            del checkpoint['state_dict']['original_fc.weight']
            del checkpoint['state_dict']['original_fc.bias']
        
        # Load state dict with strict=False to ignore size mismatches
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