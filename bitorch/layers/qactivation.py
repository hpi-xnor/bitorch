from typing import Union
from bitorch.quantizations import Quantization
import torch
from torch import nn

from bitorch.layers.layerconfig import config


class QActivation(nn.Module):
    """Activation layer for quantization"""

    def __init__(self, activation: Union[str, Quantization] = None) -> None:
        """initialization function for fetching suitable activation function.

        Args:
            activation (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.
        """
        super(QActivation, self).__init__()
        self.activation = config.get_quantization_function(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forwards input tensor through activation function.

        Args:
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized input tensor.
        """
        return self.activation(input_tensor)
