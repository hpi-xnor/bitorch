"""Module containting the quantized linear layer"""

from typing import Optional
import torch
from torch.nn import Linear
from torch.nn.functional import linear

from . import layerconfig


class QLinear(Linear):
    def __init__(self, *args, quantization: Optional[str] = None, **kwargs):  # type: ignore
        """Applys the given quantization function on weights before applying the linear operation.

        Args:
            *args (Argument list): positional arguments for linear layer
            quantization (str, optional): Name of quantization function. Defaults to None.
            **kwargs (keyword Argument list): keyword arguments for linear layer
        """
        super(QLinear, self).__init__(*args, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards x through the binary linear layer.

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """

        return linear(x, self.quantize(self.weight), self.bias)
