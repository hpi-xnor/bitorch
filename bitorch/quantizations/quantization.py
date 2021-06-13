"""Quantization superclass implementation"""

import torch
from torch import nn


class Quantization(nn.Module):
    """superclass for quantization modules"""

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """quantize the input tensor. It is recommended to use a torch.Function to also maniputlate backward behaiviour. See
        the implementations of sign or round quantization functions for more examples.

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
