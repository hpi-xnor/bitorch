"""Quantization superclass implementation"""

import torch
from torch import nn
from torch.autograd.function import Function
from typing import Any


class STE(Function):
    """Straight Through estimator for backward pass"""

    @staticmethod
    def backward(ctx: Any, output_gradient: torch.Tensor) -> torch.Tensor:
        """just passes the unchanged output gradient as input gradient.

        Args:
            ctx (Any): autograd contexxt
            output_gradient (torch.Tensor): output gradient

        Returns:
            torch.Tensor: the unchanged output gradient
        """
        return output_gradient


class Quantization(nn.Module):
    """superclass for quantization modules"""

    name = "None"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """quantize the input tensor. It is recommended to use a torch.Function to also maniputlate backward behaiviour. See
        the implementations of sign or dorefa quantization functions for more examples.

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
