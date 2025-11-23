import torch.nn as nn
import timm

class CustomSwinTransformer(nn.Module):
    def __init__(self, pretrained=True, num_classes=7):
        super(CustomSwinTransformer, self).__init__()
        self.model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)
