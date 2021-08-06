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
            bits: int = 1,
            mode: bool = True) -> torch.Tensor:
        """Quantizes the input tensor into 'bits' resolution. Depending on dorefa mode, either input quantization (mode == 1)
        or weight quantization (mode == 0) will be applied.

        Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
        Zouh et al. 2016, https://arxiv.org/abs/1606.06160


        Args:
            input_tensor (tensor): the input values to the DoReFa function
            bits (int): the bits to quantize into

        Returns:
            tensor: binarized input tensor
        """

        ctx.save_for_backward(input_tensor, torch.tensor(mode))
        max_value = 2 ** bits - 1
        if mode:
            return torch.round(torch.clamp(input_tensor, 0, 1) * max_value) / max_value

    @staticmethod
    @typing.no_type_check
    def backward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            output_grad: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """Apply straight through estimator.

        This passes the output gradient as input gradient

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the output gradient
        """
        input_tensor, mode = ctx.saved_tensors
        if mode:
            canceled_gradients = torch.logical_or(input_tensor > 1, input_tensor < 0)
            output_grad[canceled_gradients] = 0

        return output_grad, None, None


class DoReFa(Quantization):
    """Module for applying the dorefa function with straight through estimator in backward pass"""

    name = "dorefa"

    def __init__(self, bits: int = 1, mode: str = "inputs") -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to 1.
        """
        super(DoReFa, self).__init__()
        self.bits = bits
        if mode not in ["weights", "inputs"]:
            raise ValueError(f"dorefa mode must be either 'weights' or 'inputs'! Given mode: {mode}")
        self.mode = (mode == "inputs")

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        if self.mode:
            return DoReFaFunction.apply(x, self.bits, self.mode)
        else:
            max_value = 2 ** self.bits - 1
            squashed_values = torch.tanh(x)
            max_val = torch.max(abs(squashed_values))
            adjusted_values = squashed_values / (2.0 * max_val) + 0.5
            return 2.0 * (torch.round(adjusted_values * max_value) / max_value) - 1.0
