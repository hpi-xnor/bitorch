"""Sign Function Implementation"""
import torch
import typing
from typing import Tuple, Union
from torch.autograd import Function

from .base import Quantization
from .config import config


class SignFunction(Function):
    """Sign Function for input binarization."""

    @staticmethod
    def _sign(x: torch.Tensor) -> torch.Tensor:
        """Apply the sign method on the input tensor.

        Note: this function will assign 1 as the sign of 0.

        Args:
            x (torch.Tensor): the tensor to calculate the sign of

        Returns:
            torch.Tensor: the sign tensor
        """

        sign_tensor = torch.sign(x)
        return torch.where(sign_tensor == 0, torch.tensor(1., device=sign_tensor.device), sign_tensor)

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            input_tensor: torch.Tensor,
            threshold: torch.Tensor) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor, threshold)
        return SignFunction._sign(input_tensor)

    @staticmethod
    @typing.no_type_check
    def backward(
            ctx: torch.autograd.function.BackwardCFunction,  # type: ignore
            output_grad: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Apply straight through estimator.

        This passes the output gradient as input gradient after clamping the gradient values to the range [-1, 1]

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient (= the clamped output gradient)
        """
        input_tensor, threshold = ctx.saved_tensors
        return torch.where(torch.abs(input_tensor) < threshold, output_grad, torch.tensor(0., device=output_grad.device)), None


class Sign(Quantization):
    """Module for applying the sign function with straight through estimator in backward pass"""

    name = "sign"

    def __init__(self, gradient_cancelation_threshold: Union[float, None] = None) -> None:
        """Initializes gradient cancelation threshold.

        Args:
            gradient_cancelation_threshold (float, optional): threshold after which gradient is 0. Defaults to None.
        """
        super(Sign, self).__init__()
        self.gradient_cancelation_threshold = gradient_cancelation_threshold or config.gradient_cancellation_threshold
        self._threshold_tensor = None

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        if self._threshold_tensor is None:
            self._threshold_tensor = torch.tensor(
                self.gradient_cancelation_threshold, device=x.device, requires_grad=False
            )
        return SignFunction.apply(x, self._threshold_tensor)
