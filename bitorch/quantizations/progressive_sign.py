"""Progressive Sign Function"""
import typing
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from .base import Quantization, STE
from .config import config
from .sign import SignFunction

EPSILON = 1e-7


class ProgressiveSignFunctionTrain(STE):
    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Binarize the input tensor using the sign function

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
            temperature: the temperature of the incline

        Returns:
            torch.Tensor: the sign tensor
        """
        # avoid division by zero with EPSILON
        return F.hardtanh(input_tensor / max(1.0 - temperature, EPSILON))

    @staticmethod
    @typing.no_type_check
    def backward(ctx: Any, output_gradient: torch.Tensor) -> torch.Tensor:
        return output_gradient, None  # type: ignore


class ProgressiveSign(Quantization):
    """
    Module for applying a progressive sign function with STE during training.

    During validation a regular sign function is used.
    This can lead to a significant accuracy difference during the first epochs.
    With a temperature of one this function is basically equal to a regular sign function.
    """

    name = "progressive_sign"
    bit_width = 1

    scale: float
    global_scaling: bool

    def __init__(
        self,
        use_global_scaling: bool = True,
        initial_scale: Optional[float] = None,
        custom_transform: Optional[Callable[[float], float]] = None,
    ) -> None:
        """
        Initialize the progressive sign module (can be used for progressive weight binarization).

        If `use_global_scaling` is set to False, the scale of this module must be set manually.
        Otherwise, the value can be set for all progressive sign modules in the config.

        Args:
            use_global_scaling: whether to use the global scaling variable stored in the config
            initial_scale: if not using global scaling you can set an initial scale
            custom_transform: to use a custom transform function from scale to temperature, add it here
        """
        super().__init__()
        if initial_scale is not None and use_global_scaling:
            raise RuntimeWarning(
                "An initial scale was set on ProgressiveSign, but this has not effect, "
                "since use_global_scaling is True."
            )
        self.global_scaling = use_global_scaling
        self.scale = initial_scale or config.progressive_sign_scale
        self.custom_transform = custom_transform

    @property
    def current_scale(self) -> float:
        if self.global_scaling:
            return config.progressive_sign_scale
        return self.scale

    @staticmethod
    def default_transform(x: float) -> float:
        return 1 - (5 ** (-3 * x))

    def transform(self, x: float) -> float:
        """Transform x for a steady temperature increase, higher at the beginning, and much less at the end."""
        if self.custom_transform is not None:
            return self.custom_transform(x)
        return self.default_transform(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        if self.training:
            temperature = self.transform(self.current_scale)
            return ProgressiveSignFunctionTrain.apply(x, temperature)
        else:
            return SignFunction.apply(x)
