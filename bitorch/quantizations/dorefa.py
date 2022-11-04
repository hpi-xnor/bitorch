"""Dorefa Function Implementation"""
import torch
import typing
from typing import Any, Tuple, Union
from torch.autograd.function import Function

from .base import Quantization
from .config import config


class WeightDoReFaFunction(Function):
    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor, maximum_bit_value: int
    ) -> torch.Tensor:
        """quantizes input tensor and forwards it.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
            bits (int): number of bits to round the input tensor to

        Returns:
            torch.Tensor: the quantized input tensor
        """
        ctx.save_for_backward(input_tensor)

        squashed_values = torch.tanh(input_tensor)
        max_val = torch.max(torch.abs(squashed_values)).detach()
        adjusted_values = squashed_values / (2.0 * max_val) + 0.5
        return 2.0 * (torch.round(adjusted_values * maximum_bit_value) / maximum_bit_value) - 1.0

    @staticmethod
    @typing.no_type_check
    def backward(ctx: Any, output_gradient: torch.Tensor) -> torch.Tensor:
        """just passes the unchanged output gradient as input gradient.

        Args:
            ctx (Any): autograd context
            output_gradient (torch.Tensor): output gradient

        Returns:
            torch.Tensor: the unchanged output gradient
        """
        return output_gradient, None, None


class WeightDoReFa(Quantization):
    """Module for applying the dorefa function on weights.

    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    """

    name = "weightdorefa"
    bit_width = config.dorefa_bits

    def __init__(self, bits: Union[int, None] = None) -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to None.
        """
        super(WeightDoReFa, self).__init__()
        self.bit_width = bits or config.dorefa_bits
        self._max_value = 2**self.bit_width - 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution using weight dorefa.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        return WeightDoReFaFunction.apply(x, self._max_value)


class InputDoReFaFunction(Function):
    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor, bits: int  # type: ignore
    ) -> torch.Tensor:
        """quantizes input tensor and forwards it.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
            bits (int): number of bits to round the input tensor to

        Returns:
            torch.Tensor: the quantized input tensor
        """
        max_value = 2**bits - 1

        quantized_tensor = torch.round(torch.clamp(input_tensor, 0, 1) * max_value) / max_value
        return quantized_tensor

    @staticmethod
    @typing.no_type_check
    def backward(ctx: Any, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """just passes the unchanged output gradient as input gradient (i.e. applies straight through estimator)

        Args:
            ctx (Any): autograd context
            output_gradient (torch.Tensor): output gradient

        Returns:
            torch.Tensor: the unchanged output gradient
        """
        return output_gradient, None


class InputDoReFa(Quantization):
    """Module for applying the dorefa function on inputs.

    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    """

    name = "inputdorefa"
    bit_width = config.dorefa_bits

    def __init__(self, bits: Union[int, None] = None) -> None:
        """Initiates quantization bits.

        Args:
            bits (int, optional): number of bits to quantize into. Defaults to None.
        """
        super(InputDoReFa, self).__init__()
        self.bit_width = bits or config.dorefa_bits

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        return InputDoReFaFunction.apply(x, self.bit_width)
