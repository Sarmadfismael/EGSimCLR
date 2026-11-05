import torch
import torch.nn as nn
import torchvision.models as models
import config


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet50": models.resnet50(weights=None, num_classes=out_dim),}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # MLP projection head (SimCLR style)
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

        # ✅ Freeze encoder if needed
        if config.FREEZE_ENCODER:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Choose one of: 'resnet18' or 'resnet50'"
            )
        return model

    def forward(self, x):
        return self.backbone(x)
