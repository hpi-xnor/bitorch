"""Dorefa Function Implementation"""
import torch
from typing import Union

from .base import Quantization, STE
from .config import config


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
        self._max_value = 2 ** self.bits - 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution using weight dorefa.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """
        squashed_values = torch.tanh(x)
        max_val = torch.max(torch.abs(squashed_values)).detach()
        adjusted_values = squashed_values / (2.0 * max_val) + 0.5
        return 2.0 * (torch.round(adjusted_values * self._max_value) / self._max_value) - 1.0


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
        self._max_value = 2 ** self.bits - 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """DoReFas the tensor to desired bit resolution.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: DoReFaed tensor x
        """

        return STE.apply(torch.round(torch.clamp(x, 0, 1) * self._max_value) / self._max_value)
