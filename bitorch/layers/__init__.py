"""
This submodule contains adapted pytorch layers that use quantization functions on their weights
and activations before forwarding them. These layers use the quantization functions specified in the
quantization submodule.
"""

from .debug_layers import (
    InputGraphicalDebug,
    InputPrintDebug,
    WeightGraphicalDebug,
    WeightPrintDebug,
    ShapePrintDebug
)
from .qactivation import QActivation
from .qconv1d import QConv1d, QConv1d_NoAct
from .qconv2d import QConv2d, QConv2d_NoAct
from .qconv3d import QConv3d, QConv3d_NoAct
from .qlinear import QLinear
from .pact import Pact
from .qembedding import QEmbedding, QEmbeddingBag

__all__ = [
    "InputGraphicalDebug", "InputPrintDebug", "WeightGraphicalDebug", "WeightPrintDebug",
    "ShapePrintDebug", "QActivation", "QConv1d", "QConv2d", "QConv3d", "QConv1d_NoAct",
    "QConv2d_NoAct", "QConv3d_NoAct", "QLinear", "QEmbedding", "QEmbeddingBag", "Pact",
]
