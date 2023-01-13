"""Sign Function Implementation"""

import typing

import torch

from .base import Quantization, STE


class SignFunction(STE):
    @staticmethod
    @typing.no_type_check
    def forward(
        ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Binarize the input tensor using the sign function.

        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the sign tensor
        """
        sign_tensor = torch.sign(input_tensor)
        sign_tensor = torch.where(sign_tensor == 0, torch.tensor(1.0, device=sign_tensor.device), sign_tensor)
        return sign_tensor


class Sign(Quantization):
    """Module for applying the sign function with straight through estimator in backward pass."""

    name = "sign"
    bit_width = 1

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SignFunction.apply(x)
