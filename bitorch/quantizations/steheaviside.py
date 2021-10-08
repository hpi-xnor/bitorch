"""Sign Function Implementation"""
import torch
from .base import Quantization


class SteHeaviside(Quantization):
    """Module for applying the SteHeaviside quantization, using an ste in backward pass"""

    name = "steheaviside"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the tensor through the sign function.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: sign of tensor x
        """
        sign_tensor = torch.sign(x)
        sign_tensor[sign_tensor < 0] = 0
        return sign_tensor
