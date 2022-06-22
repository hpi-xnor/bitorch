"""Sign Function Implementation"""

import torch
from torch.autograd.function import Function
import typing
from typing import Tuple, Union

from .base import Quantization
from .config import config


class SwishSignFunction(Function):
    """SwishSign Function for input binarization."""

    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor, torch.tensor(beta, device=input_tensor.device))

        sign_tensor = torch.sign(input_tensor)
        sign_tensor = torch.where(sign_tensor == 0, torch.tensor(1.0, device=input_tensor.device), sign_tensor)
        return sign_tensor

    @staticmethod
    @typing.no_type_check
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, output_grad: torch.Tensor  # type: ignore
    ) -> Tuple[torch.Tensor, None]:
        """Apply straight through estimator.

        This passes the output gradient as input gradient after clamping the gradient values to the range [-1, 1]

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient (= the clamped output gradient)
        """
        input_tensor, beta = ctx.saved_tensors
        # produces zeros where preactivation inputs exceeded threshold, ones otherwise
        swish = (beta * (2 - beta * input_tensor * torch.tanh(beta * input_tensor / 2))) / (
            1 + torch.cosh(beta * input_tensor)
        )
        return swish * output_grad, None


class SwishSign(Quantization):
    """Module for applying the SwishSign function"""

    name = "swishsign"
    bit_width = 1

    def __init__(self, beta: Union[float, None] = None) -> None:
        """Initializes gradient cancelation threshold.

        Args:
            gradient_cancelation_threshold (float, optional): threshold after which gradient is 0. Defaults to 1.0.
        """
        super(SwishSign, self).__init__()
        self.beta = beta or config.beta

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the swishsign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SwishSignFunction.apply(x, self.beta)
