"""Sign Function Implementation"""
import torch
from torch.autograd.function import Function
import typing

from .base import Quantization


class ApproxSignFunction(Function):
    """ApproxSign Function for input binarization."""

    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor  # type: ignore
    ) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor)

        sign_tensor = torch.sign(input_tensor)
        sign_tensor = torch.where(sign_tensor == 0, torch.tensor(1.0, device=sign_tensor.device), sign_tensor)
        return sign_tensor

    @staticmethod
    @typing.no_type_check
    def backward(
        ctx: torch.autograd.function.BackwardCFunction, output_grad: torch.Tensor  # type: ignore
    ) -> torch.Tensor:
        """Apply approx sign function. used e.g. for birealnet

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient
        """
        input_tensor = ctx.saved_tensors[0]
        # produces zeros where preactivation inputs exceeded threshold, ones otherwise
        inside_threshold = torch.abs(input_tensor) <= 1
        approx_sign = (2.0 - 2.0 * torch.abs(input_tensor)) * inside_threshold
        return approx_sign * output_grad


class ApproxSign(Quantization):
    """Module for applying the sign function with approx sign in backward pass"""

    name = "approxsign"
    bit_width = 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the approx sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return ApproxSignFunction.apply(x)  # type: ignore
