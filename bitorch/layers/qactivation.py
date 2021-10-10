import typing
from typing import Union, Tuple
import torch
from torch import nn
from torch.autograd.function import Function

from bitorch.quantizations import Quantization
from .config import config


class GradientCancellation(Function):

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
        ctx.save_for_backward(input_tensor, torch.tensor(threshold, device=input_tensor.device))
        return input_tensor

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
        cancelled = torch.where(
            torch.abs(input_tensor) < threshold,
            output_grad,
            torch.tensor(1., device=output_grad.device))
        return cancelled, None


class QActivation(nn.Module):
    """Activation layer for quantization"""

    def __init__(
            self,
            activation: Union[str, Quantization] = None,
            gradient_cancellation_threshold: Union[float, None] = None) -> None:
        """initialization function for fetching suitable activation function.

        Args:
            activation (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.
        """
        super(QActivation, self).__init__()
        self._activation = config.get_quantization_function(activation)
        self._gradient_cancellation_threshold = (
            gradient_cancellation_threshold or config.gradient_cancellation_threshold
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forwards input tensor through activation function.

        Args:
            input_tensor (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized input tensor.
        """
        activated_input = self._activation(input_tensor)
        if self._gradient_cancellation_threshold:
            return GradientCancellation.apply(activated_input, self._gradient_cancellation_threshold)
        else:
            return activated_input
