from torch import nn
import torch


class TorchLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, preds, target):
        return self.loss(preds, target)