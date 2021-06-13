"""Module containting the quantized linear layer"""

from typing import Union
import torch
from torch.nn import Linear
from torch.nn.functional import linear

from bitorch.quantizations import Quantization
from . import layerconfig


class QLinear(Linear):
    def __init__(self, *args, weight_quantization: Union[str, Quantization] = None, **kwargs):  # type: ignore
        """Applys the given quantization function on weights before applying the linear operation.

        Args:
            *args (Argument list): positional arguments for linear layer
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function. Defaults to None.
            **kwargs (keyword Argument list): keyword arguments for linear layer
        """
        super(QLinear, self).__init__(*args, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(weight_quantization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards x through the binary linear layer.

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """

        return linear(x, self.quantize(self.weight), self.bias)
