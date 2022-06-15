"""
This submodule contains adapted pytorch layers that use quantization functions on their weights
and activations before forwarding them. These layers use the quantization functions specified in the
quantization submodule.
"""
from typing import List, TypeVar

import torch
from torch import nn

from .debug_layers import (
    InputGraphicalDebug,
    InputPrintDebug,
    WeightGraphicalDebug,
    WeightPrintDebug,
    ShapePrintDebug
)
from .extensions import CustomImplementation, LayerRegistry
from .pact import Pact
from .qactivation import QActivation
from .qconv1d import QConv1d, QConv1d_NoAct, q_conv1d_registry
from .qconv2d import QConv2d, QConv2d_NoAct, q_conv2d_registry
from .qconv3d import QConv3d, QConv3d_NoAct, q_conv3d_registry
from .qembedding import QEmbedding, QEmbeddingBag
from .qlinear import QLinear, QLinearBase, q_linear_registry

__all__ = [
    "InputGraphicalDebug", "InputPrintDebug", "WeightGraphicalDebug", "WeightPrintDebug",
    "ShapePrintDebug", "QActivation", "QConv1d", "QConv2d", "QConv3d", "QConv1d_NoAct",
    "QConv2d_NoAct", "QConv3d_NoAct", "QLinear", "QLinearBase", "QEmbedding", "QEmbeddingBag", "Pact",
    "CustomImplementation", "convert"
]

from .. import RuntimeMode


def _get_layer_registries() -> List[LayerRegistry]:
    return [
        q_conv1d_registry,
        q_conv2d_registry,
        q_conv3d_registry,
        q_linear_registry,
    ]


T = TypeVar("T", bound=nn.Module)


def convert(module: T, new_mode: RuntimeMode, device: torch.device = None, verbose: bool = False) -> T:
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
    for registry in _get_layer_registries():
        registry.convert_layers_to(new_mode, only=submodules, device=device, verbose=verbose)
    module.to(device)
    return module
