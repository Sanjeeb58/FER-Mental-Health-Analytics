import torch.nn as nn
import timm

class CustomViT(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(CustomViT, self).__init__()
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
