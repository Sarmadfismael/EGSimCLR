from torchvision.models import resnet50
import torch.nn as nn
import config

def get_model():
    # Load pretrained ResNet-50
    model = resnet50(weights=None)

    # Replace the fully connected layer (classifier)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)

    return model
