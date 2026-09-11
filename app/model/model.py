import torch.nn as nn
import timm

class WeedNet(nn.Module):

    def __init__(self, n_classes: int, drop: float = 0.3):
        super().__init__()

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )

        d = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Dropout(drop),
            nn.Linear(d, 512),
            nn.GELU(),
            nn.Dropout(drop / 2),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))