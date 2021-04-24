"""Module containting the quantized dense layer"""

import torch
from torch.nn import Linear
from torch.nn.functional import linear

from . import layerconfig


class QDense(Linear):
    def __init__(self, *kargs, quantization: str = None, **kwargs):  # type: ignore
        """Applys the given quantization function on weights before applying the linear operation.

        Args:
            quantization (str, optional): Name of quantization function. Defaults to None.
        """
        super(QDense, self).__init__(*kargs, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards x through the binary linear layer.

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """

        return linear(x, self.quantize(self.weight), self.bias)
