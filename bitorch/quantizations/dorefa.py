"""Dorefa Function Implementation"""

import typing
import torch
from typing import Tuple

from torch.autograd.function import Function
from .base import Quantization


class DoReFaFunction(Function):
    """DoReFa Function for input quantization. Uses STE for backward pass"""

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor,
            bits: int = 1) -> torch.Tensor:
        """Quantizes the input tensor into 'bits' resolution

        Args:
            input_tensor (tensor): the input values to the DoReFa function
            bits (int): the bits to quantize into

        Returns:
            tensor: binarized input tensor
        """
        max_value = 2 ** bits - 1
        return torch.round(torch.clamp(input_tensor, 0, 1) * max_value) / max_value

    @staticmethod
    @typing.no_type_check
    def backward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            output_grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Apply straight through estimator.

        This passes the output gradient as input gradient

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the output gradient
        """
        return output_grad, None


class DoReFa(Quantization):
    """Module for applying the dorefa function with straight through estimator in backward pass"""

    name = "dorefa"

    def __init__(self, bits: int = 1) -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to 1.
        """
        super(DoReFa, self).__init__()
        self.bits = bits

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        return DoReFaFunction.apply(x, self.bits)
