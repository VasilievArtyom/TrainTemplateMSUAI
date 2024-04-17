import timm
from torch import nn


class TorchModel(nn.Module):
    def __init__(self, encoder, output):
        super(TorchModel, self).__init__()
        self.model = timm.create_model(encoder, pretrained=True, num_classes=output)

    def forward(self, x):
        return self.model(x)
