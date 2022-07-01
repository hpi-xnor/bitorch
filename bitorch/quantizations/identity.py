"""Identity Implementation"""

from .base import Quantization
import torch


class Identity(Quantization):
    """Module that provides the identity function, which can be useful for certain training strategies"""

    name = "identity"
    bit_width = 32

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x without quantization.

        Args:
            x (torch.Tensor): tensor to be forwarded.

        Returns:
            torch.Tensor: tensor x
        """
        return x
