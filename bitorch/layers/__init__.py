"""
This submodule contains adapted pytorch layers that use quantization functions on their weights
and activations before forwarding them. These layers use the quantization functions specified in the
quantization submodule.
"""
from typing import Optional, TypeVar

import torch
from torch import nn

from bitorch import RuntimeMode
from .debug_layers import InputGraphicalDebug, InputPrintDebug, WeightGraphicalDebug, WeightPrintDebug, ShapePrintDebug
from .extensions import CustomImplementationMixin
from .pact import Pact
from .qactivation import QActivation
from .qconv1d import QConv1d, QConv1dBase, QConv1d_NoAct
from .qconv2d import QConv2d, QConv2dBase, QConv2d_NoAct
from .qconv3d import QConv3d, QConv3dBase, QConv3d_NoAct
from .qembedding import QEmbedding, QEmbeddingBag
from .qlinear import QLinear, QLinearBase
from .register import all_layer_registries
from .pad import PadModule

__all__ = [
    "InputGraphicalDebug",
    "InputPrintDebug",
    "WeightGraphicalDebug",
    "WeightPrintDebug",
    "ShapePrintDebug",
    "QActivation",
    "QConv1d",
    "QConv2d",
    "QConv3d",
    "QConv1dBase",
    "QConv2dBase",
    "QConv3dBase",
    "QConv1d_NoAct",
    "QConv2d_NoAct",
    "QConv3d_NoAct",
    "QLinear",
    "QLinearBase",
    "QEmbedding",
    "QEmbeddingBag",
    "Pact",
    "CustomImplementationMixin",
    "convert",
    "PadModule",
]


T = TypeVar("T", bound=nn.Module)


def convert(module: T, new_mode: RuntimeMode, device: Optional[torch.device] = None, verbose: bool = False) -> T:
    """
    Convert the given module to a new bitorch RuntimeMode. Needs to have custom implementations installed.

    Args:
        module: the module to be converted
        new_mode: the new mode for the module
        device: an optional device
        verbose: whether to print which layers are converted

    Returns:
        the converted module
    """
    submodules = list(module.modules())
    for registry in all_layer_registries():
        registry.convert_layers_to(new_mode, only=submodules, device=device, verbose=verbose)
    return module
