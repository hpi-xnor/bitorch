"""Sign Function Implementation"""

from torch import tensor
from .base import Quantization
import torch
import typing

from .sign import SignFunction


class SteHeavisideFunction(SignFunction):
    """SteHeaviside Function for input binarization. the backward pass uses an Straight through estimator defined in
    SignFunction superclass."""

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor,
            threshold: float = 1.0) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor, torch.tensor(threshold))
        sign_tensor = torch.sign(tensor)
        sign_tensor[sign_tensor < 0] = 0
        return sign_tensor


class SteHeaviside(Quantization):
    """Module for applying the SteHeaviside quantization, using an ste in backward pass"""

    name = "steheaviside"

    def __init__(self, gradient_cancelation_threshold: float = 1.0) -> None:
        """Initializes gradient cancelation threshold.

        Args:
            gradient_cancelation_threshold (float, optional): threshold after which gradient is 0. Defaults to 1.0.
        """
        super(SteHeaviside, self).__init__()
        self.gradient_cancelation_threshold = gradient_cancelation_threshold

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        return SteHeaviside.apply(x, self.gradient_cancelation_threshold)  # type: ignore
