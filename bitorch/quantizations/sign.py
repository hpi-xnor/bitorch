"""Sign Function Implementation"""
import torch
from .base import Quantization


class Sign(Quantization):
    """Module for applying the sign function with straight through estimator in backward pass"""

    name = "sign"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        sign_tensor = torch.sign(x)
        sign_tensor[sign_tensor == 0] = 1
        return sign_tensor
