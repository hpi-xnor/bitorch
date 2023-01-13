"""Module containing the quantized linear layer"""

from typing import Optional, Any, Type, Union, Dict

import torch
from torch.nn import Linear
from torch.nn.functional import linear

from bitorch import RuntimeMode
from bitorch.quantizations import Quantization
from .config import config
from .extensions import LayerRecipe, DefaultImplementationMixin
from .qactivation import QActivation
from .register import QLinearImplementation


class QLinearBase(Linear):
    def __init__(
        self,
        *args: int,
        input_quantization: Optional[Union[str, Quantization]] = None,
        gradient_cancellation_threshold: Union[float, None] = None,
        weight_quantization: Optional[Union[str, Quantization]] = None,
        **kwargs: bool,
    ) -> None:
        """Applies the given quantization functions on weights and inputs before applying the linear operation.

        Args:
            *args: positional arguments for linear layer
            input_quantization (Union[str, Quantization], optional): quantization module used for input
                quantization. Defaults to None.
            gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient cancellation.
                disabled if threshold is None. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function. Defaults to None.
            **kwargs: keyword arguments for linear layer
        """
        super().__init__(*args, **kwargs)  # type: ignore
        self.weight_quantization = config.get_quantization_function(weight_quantization or config.weight_quantization)
        self.activation = QActivation(input_quantization, gradient_cancellation_threshold)

    @staticmethod
    def get_args_as_kwargs(recipe: LayerRecipe) -> Dict[str, Any]:
        """
        Gather all arguments that were used to create a QLinear layer with argument names.
        Can be used to recreate a layer with identical arguments.

        Returns:
            A dictionary with all arguments (key is the argument name as a string even for positional arguments)
        """
        return {
            "in_features": recipe.get_positional_arg(0),
            "out_features": recipe.get_positional_arg(1),
            "input_quantization": recipe.layer.input_quantization,
            "gradient_cancellation_threshold": recipe.layer.gradient_cancellation_threshold,
            "weight_quantization": recipe.layer.weight_quantization,
            "bias": recipe.get_arg(5, "bias", True),
            "device": recipe.get_arg(6, "device", None),
            "dtype": recipe.get_arg(7, "dtype", None),
        }

    @property
    def input_quantization(self) -> Quantization:
        return self.activation.activation_function

    @property
    def gradient_cancellation_threshold(self) -> float:
        return self.activation.gradient_cancellation_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards x through the binary linear layer.

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """
        return linear(self.activation(x), self.weight_quantization(self.weight), self.bias)


class _QLinearComposed(DefaultImplementationMixin, QLinearBase):
    """
    This class defines the default implementation of a QLinear layer (which is actually implemented by QLinearBase).

    To implement a custom QLinear implementation use QLinearBase as a super class instead.
    """

    pass


QLinear: Type[_QLinearComposed] = QLinearImplementation(RuntimeMode.DEFAULT)(_QLinearComposed)  # type: ignore
"""
This class provides the current implementation of a QLinear layer (which is actually implemented by :class:`QLinearBase`).

To implement a custom QLinear implementation use :class:`QLinearBase` as a super class instead.
"""
