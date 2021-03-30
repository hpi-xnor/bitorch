"""Sign Function Implementation"""

import torch
from torch.autograd import Function


class SignFunction(Function):
    """Sign Function for input binarization."""

    @staticmethod
    def _sign(tensor):
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
    def forward(ctx, input_tensor):
        """Binarize input tensor using the _sign function.

        Args:
            input_tensor (tensor): the input values to the Sign function

        Returns:
            tensor: binarized input tensor
        """
        return SignFunction._sign(input_tensor)

    @staticmethod
    def backward(ctx, output_grad):
        """Apply straight through estimator.

        This passes the output gradient as input gradient after clamping the gradient values to the range [-1, 1]

        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient

        Returns:
            torch.Tensor: the input gradient (= the clamped output gradient)
        """
        return torch.clamp(output_grad, -1, 1)
