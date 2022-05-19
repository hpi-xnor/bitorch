"""Module containting the quantized linear layer"""

from typing import Union
import torch
from torch.nn import Linear
from torch.nn.functional import linear

from bitorch.quantizations import Quantization
from .config import config
from .qactivation import QActivation


class QLinear(Linear):
    def __init__(
            self,
            *args: int,
            input_quantization: Union[str, Quantization] = None,
            gradient_cancellation_threshold: Union[float, None] = None,
            weight_quantization: Union[str, Quantization] = None,
            **kwargs: bool) -> None:
        """Applys the given quantization functions on weights and inputs before applying the linear operation.

        Args:
            *args (Argument list): positional arguments for linear layer
            input_quantization (Union[str, Quantization], optional): quantization module used for input
                quantization. Defaults to None.
            gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient cancellation.
                disabled if threshold is None. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function. Defaults to None.
            **kwargs (keyword Argument list): keyword arguments for linear layer
        """

        super(QLinear, self).__init__(*args, **kwargs)  # type: ignore
        self.weight_quantize = config.get_quantization_function(weight_quantization or config.weight_quantization)
        self.activation = QActivation(input_quantization, gradient_cancellation_threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards x through the binary linear layer.

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """

        return linear(self.activation(x), self.weight_quantize(self.weight), self.bias)
