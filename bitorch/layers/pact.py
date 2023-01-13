from typing import Optional, Tuple
from torch.autograd import Function
from torch.nn import Module
import torch

from .config import config


# Taken from:
# https://github.com/KwangHoonAn/PACT
class PactActFn(Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, alpha: torch.nn.Parameter, bits: int) -> torch.Tensor:  # type: ignore
        ctx.save_for_backward(input_tensor, alpha)
        # y_1 = 0.5 * ( torch.abs(x).detach() - torch.abs(x - alpha).detach() + alpha.item() )
        clamped = torch.clamp(input_tensor, min=0, max=alpha.item())
        scale = (2**bits - 1) / alpha
        quantized = torch.round(clamped * scale) / scale
        return quantized

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:  # type: ignore
        # Backward function, I borrowed code from
        # https://github.com/obilaniu/GradOverride/blob/master/functional.py
        # We get dL / dy_q as a gradient
        x, alpha = ctx.saved_tensors
        # Weight gradient is only valid when [0, alpha]
        # Actual gradient for alpha,
        # By applying Chain Rule, we get dL / dy_q * dy_q / dy * dy / dalpha
        # dL / dy_q = argument,  dy_q / dy * dy / dalpha = 0, 1 with x value range
        lower_bound = x < 0
        upper_bound = x > alpha
        # x_range       = 1.0-lower_bound-upper_bound
        x_range = ~(lower_bound | upper_bound)
        grad_alpha = torch.sum(output_gradient * torch.ge(x, alpha).float()).view(-1)
        return output_gradient * x_range.float(), grad_alpha, None


class Pact(Module):
    """Pact activation function taken from https://github.com/KwangHoonAn/PACT.
    Initially proposed in
    Choi, Jungwook, et al. "Pact: Parameterized clipping activation for quantized neural networks." (2018)
    """

    def __init__(self, bits: Optional[int] = None) -> None:
        super().__init__()
        self.alpha = torch.nn.parameter.Parameter(torch.tensor(10.0))
        self.bits = bits or config.pact_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return PactActFn.apply(x, self.alpha, self.bits)
