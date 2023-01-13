"""Quantization superclass implementation"""

import typing
from typing import Any
from warnings import warn

import torch
from torch import nn
from torch.autograd.function import Function


class STE(Function):
    """Straight Through estimator for backward pass"""

    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError("Forwards pass of STE should be implemented by subclass.")

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
        return output_gradient


class Quantization(nn.Module):
    """superclass for quantization modules"""

    name: str = "None"
    bit_width: int = -1

    @property
    def bitwidth(self) -> int:
        warn("Attribute 'bitwidth' is deprecated, use 'bit_width' instead.", DeprecationWarning, stacklevel=2)
        return self.bit_width

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the quantization function to the input tensor.

        It is recommended to use a torch.Function to also manipulate backwards behavior.
        See the implementations of sign or dorefa quantization functions for more examples.

        Args:
            x (torch.Tensor): the input to be quantized

        Raises:
            NotImplementedError: raised if quantize function of superclass is called.

        Returns:
            torch.Tensor: the quantized tensor
        """
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantizes the tensor using this classes quantize-method. Subclasses shall add some semantic there.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: quantized tensor x
        """
        return self.quantize(x)
