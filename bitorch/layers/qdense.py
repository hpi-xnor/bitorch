"""Module containting the quantized dense layer"""

from torch.nn import Linear
from torch.nn import Module

from . import qactivation
from . import layerconfig


class QDense(Module):
    def __init__(self, input_size, output_size):
        self.linear = Linear(3, 4)
        self.activation = qactivation()

    def forward(self, x):
        return self.linear(x)
