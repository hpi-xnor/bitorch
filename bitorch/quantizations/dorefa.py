"""Dorefa Function Implementation"""

import typing
import torch
from typing import Tuple, Union
from torch.autograd.function import Function

from .base import Quantization
from .config import config


class DoReFaFunction(Function):
    """DoReFa Function for input quantization. Uses STE for backward pass"""

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor,
            bits: int = 1) -> torch.Tensor:
        """Quantizes the input tensor into 'bits' resolution.

        Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
        Zouh et al. 2016, https://arxiv.org/abs/1606.06160


        Args:
            input_tensor (tensor): the input values to the DoReFa function
            bits (int): the bits to quantize into

        Returns:
            tensor: binarized input tensor
        """

        ctx.save_for_backward(input_tensor)
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
        input_tensor = ctx.saved_tensors[0]
        print(input_tensor)
        canceled_gradients = torch.logical_or(input_tensor > 1, input_tensor < 0)
        output_grad[canceled_gradients] = 0

        return output_grad, None


class WeightDoReFa(Quantization):
    """Module for applying the dorefa function on weights.

    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    """

    name = "weightdorefa"

    def __init__(self, bits: int = 1) -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to 1.
        """
        super(WeightDoReFa, self).__init__()
        self.bits = bits

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution using weight dorefa.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        max_value = 2 ** self.bits - 1
        squashed_values = torch.tanh(x)
        max_val = torch.max(torch.abs(squashed_values)).detach()
        adjusted_values = squashed_values / (2.0 * max_val) + 0.5
        return 2.0 * (torch.round(adjusted_values * max_value) / max_value) - 1.0


class InputDoReFa(Quantization):
    """Module for applying the dorefa function on inputs.

    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    """

    name = "inputdorefa"

    def __init__(self, bits: Union[int, None] = 1) -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to 1.
        """
        super(InputDoReFa, self).__init__()
        self.bits = bits or config.dorefa_bits

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        return DoReFaFunction.apply(x, self.bits)
