import torch
from torch import nn

from . import layerconfig


class QActivation(nn.Module):
    """Activation layer for quantization"""

    def __init__(self, activation: str = None) -> None:
        """initialization function for fetching suitable activation function.

        Args:
            activation (str, optional): name of quantization function. Defaults to None.
        """
        super(QActivation, self).__init__()
        self.activation = layerconfig.config.get_quantization_function(activation)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forwards input tensor through activation function.

        Args:
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized input tensor.
        """
        # grad cancel on every quantization function? always ste?
        return self.activation(input_tensor)
