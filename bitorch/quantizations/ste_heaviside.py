"""Sign Function Implementation"""
import torch
import typing
from typing import Any
from .base import STE, Quantization


class SteHeavisideFunction(STE):
    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """quantizes input tensor and forwards it.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the quantized input tensor
        """
        ctx.save_for_backward(input_tensor)

        quantized_tensor = torch.where(
            input_tensor > 0,
            torch.tensor(1.0, device=input_tensor.device),
            torch.tensor(-1.0, device=input_tensor.device),
        )
        return quantized_tensor

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
        input_tensor = ctx.saved_tensors[0]
        inside_threshold = torch.abs(input_tensor) <= 1
        print("over threshold:", len(input_tensor) - torch.sum(inside_threshold))
        return output_gradient * inside_threshold


class SteHeaviside(Quantization):
    """Module for applying the SteHeaviside quantization, using an ste in backward pass"""

    name = "steheaviside"
    bit_width = 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SteHeavisideFunction.apply(x)
