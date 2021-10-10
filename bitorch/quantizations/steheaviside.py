"""Sign Function Implementation"""
import torch
from .base import STE, Quantization


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
        return STE.apply(torch.where(x >= 0, torch.tensor(1., device=x.device), x))
