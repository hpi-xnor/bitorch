"""Sign Function Implementation"""

from .base import Quantization
from typing import Tuple
import torch
import typing
from torch.autograd import Function
from .sign import SignFunction


class ApproxSignFunction(SignFunction):
    """ApproxSign Function for input binarization."""

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor)
        return ApproxSignFunction._sign(input_tensor)

    @staticmethod
    @typing.no_type_check
    def backward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            output_grad: torch.Tensor) -> torch.Tensor:
        """Apply approx sign function. used e.g. for birealnet

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient
        """
        input_tensor = ctx.saved_tensors
        # produces zeros where preactivation inputs exceeded threshold, ones otherwise
        inside_threshold = (torch.abs(input_tensor) <= 1)
        approx_sign = (2.0 - 2.0 * torch.aps(input_tensor)) * inside_threshold
        return approx_sign * output_grad


class ApproxSign(Quantization):
    """Module for applying the sign function with approx sign in backward pass"""

    name = "approxsign"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the approx sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return ApproxSign.apply(x)
