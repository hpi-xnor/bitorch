"""Sign Function Implementation"""
import torch
import typing
from .base import STE, Quantization


class SteHeavisideFunction(STE):
    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor) -> torch.Tensor:
        """quantizes input tensor and forwards it.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the quantized input tensor
        """

        quantized_tensor = torch.where(input_tensor > 0, torch.tensor(
            1., device=input_tensor.device), torch.tensor(0., device=input_tensor.device))
        return quantized_tensor


class SteHeaviside(Quantization):
    """Module for applying the SteHeaviside quantization, using an ste in backward pass"""

    name = "steheaviside"
    bitwidth = 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SteHeavisideFunction.apply(x)
