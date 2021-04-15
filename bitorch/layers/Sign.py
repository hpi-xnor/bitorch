"""Sign Function Implementation"""

from typing import Tuple
import torch
from torch.autograd import Function
from torch import nn
import typing


class SignFunction(Function):
    """Sign Function for input binarization."""

    @staticmethod
    def _sign(tensor: torch.Tensor) -> torch.Tensor:
        """Apply the sign method on the input tensor.

        Note: this function will assign 1 as the sign of 0.

        Args:
            tensor (torch.Tensor): the tensor to calculate the sign of

        Returns:
            torch.Tensor: the sign tensor
        """

        sign_tensor = torch.sign(tensor)
        sign_tensor[sign_tensor == 0] = 1
        return sign_tensor

    @staticmethod
    @typing.no_type_check
    def forward(
            ctx: torch.autograd.function.BackwardCFunction,
            input_tensor: torch.Tensor,
            threshold: float = 1.0) -> torch.Tensor:
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        ctx.save_for_backward(input_tensor, torch.tensor(threshold))
        return SignFunction._sign(input_tensor)

    @staticmethod
    @typing.no_type_check
    def backward(
            ctx: torch.autograd.function.BackwardCFunction,
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
        # produces zeros where preactivation inputs exceeded threshold, ones otherwise
        input_grad = (torch.abs(input_tensor) < threshold)
        return input_grad * output_grad, None


class Sign(nn.Module):
    def __init__(self) -> None:
        super(Sign, self).__init__()

    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        return SignFunction.apply(x, t)
