from torch import nn, Tensor
import torch.nn.functional as F


class PadModule(nn.Module):
    """Module for padding tensors."""

    def __init__(
        self,
        padding_left: int = 0,
        padding_right: int = 0,
        padding_top: int = 0,
        padding_bottom: int = 0,
        padding_value: int = 0,
    ):
        """initialization function for padding.

        Args:
            padding_left (int, optional): number of columns to pad to the left.
            padding_right (int, optional): number of columns to pad to the right.
            padding_top (int, optional): number of rows to pad at the top.
            padding_bottom (int, optional): number of rows to pad at the bottom.
            padding_value (float, optional): fill value used for padding.
        """
        super(PadModule, self).__init__()
        self.padding_tensor = (padding_left, padding_right, padding_top, padding_bottom)
        self.padding_value = padding_value

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, self.padding_tensor, "constant", self.padding_value)
        return x
