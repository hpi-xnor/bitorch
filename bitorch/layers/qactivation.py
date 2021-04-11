import torch
from torch import nn

from . import layerconfig


class QActivation(nn.Module):
    def __init__(self) -> None:
        super(QActivation, self).__init__()
        self.quantization = layerconfig.config.quantization

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.quantization(input_tensor)
