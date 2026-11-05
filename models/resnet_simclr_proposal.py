import torch
import torch.nn as nn
import torchvision.models as models
import config

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, k_hat=25):  # Default to 36 to match pre-trained model
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet50": models.resnet50(weights=None),
        }

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # Remove the original_fc reference since it's causing issues
        # Just replace the FC layer directly
        self.backbone.fc = nn.Identity()

        # MLP projection head (SimCLR style)
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

        # Classifier head for entropy calculation
        self.classifier_head = nn.Sequential(
            nn.Linear(dim_mlp, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, k_hat),
        )

        # Freeze encoder if needed
        if config.FREEZE_ENCODER:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(
                "Invalid backbone architecture. Choose one of: 'resnet50'"
            )
        return model

    def forward(self, x, return_features=False):
        # Extract features from the backbone
        features = self.backbone(x)
        
        # Get projection for contrastive learning
        projection = self.projection_head(features)
        
        # Get class logits for entropy calculation
        logits = self.classifier_head(features)
        
        if return_features:
            return features, projection, logits
        return projection, logits

    def get_encoder(self):
        return self.backbone