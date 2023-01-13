"""Progressive Sign Function"""
import typing
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch.autograd.function import Function

from bitorch.config import Config
from .base import Quantization
from .sign import SignFunction

EPSILON = 1e-7


class ProgressiveSignConfig(Config):
    name = "progressive_sign_config"

    # scaling of progressive sign function, should be zero at the start of the training, and (close to) one at the end
    progressive_sign_scale = 0.0

    # alpha of default progressive sign transform function, should be between 2 and 10
    progressive_sign_alpha = 4

    # beta of default progressive sign transform function, should be between 2 and 10
    progressive_sign_beta = 10


config = ProgressiveSignConfig()


class ProgressiveSignFunctionTrain(Function):
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
        ctx.save_for_backward(input_tensor)
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
    alpha: Union[int, float]
    beta: Union[int, float]

    def __init__(
        self,
        use_global_scaling: bool = True,
        initial_scale: Optional[float] = None,
        custom_transform: Optional[Callable[[float], float]] = None,
        alpha: Optional[Union[int, float]] = None,
        beta: Optional[Union[int, float]] = None,
    ) -> None:
        """
        Initialize the progressive sign module (can be used for progressive weight binarization).

        If `use_global_scaling` is set to False, the scale of this module must be set manually.
        Otherwise, the value can be set for all progressive sign modules in the config.

        Args:
            use_global_scaling: whether to use the global scaling variable stored in the config
            initial_scale: if not using global scaling you can set an initial scale
            custom_transform: to use a custom transform function from scale to temperature, add it here
            alpha: parameters of default transform function
            beta: parameters of default transform function
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
        self.alpha = alpha or config.progressive_sign_alpha
        self.beta = beta or config.progressive_sign_beta

    @property
    def current_scale(self) -> float:
        """Return the current scale of this Progressive Sign layer."""
        if self.global_scaling:
            return config.progressive_sign_scale
        return self.scale

    @staticmethod
    def default_transform(
        scale: float, alpha: Optional[Union[int, float]] = None, beta: Optional[Union[int, float]] = None
    ) -> float:
        """Transform the given scale into the temperature of the progressive sign function with the default function.

        The formula is as follows: 1 - (alpha ** (-beta * scale))

        Args:
            scale: the current scale
            alpha: base of default exponential function
            beta: (negative) factor of scale exponent
        """
        if alpha is None:
            alpha = config.progressive_sign_alpha
        if beta is None:
            beta = config.progressive_sign_beta
        return 1 - (alpha ** (-beta * scale))

    def transform(self, scale: float) -> float:
        """Transform the given scale into a steady temperature increase, higher at the start, and much less at the end.

        Args:
            scale: the current scale
        """
        if self.custom_transform is not None:
            return self.custom_transform(scale)
        return self.default_transform(scale, self.alpha, self.beta)

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
