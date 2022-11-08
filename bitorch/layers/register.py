from typing import List, Iterable, Any, Optional

import torch

from bitorch import runtime_mode_type, RuntimeMode
from bitorch.layers.extensions import LayerImplementation, LayerRegistry

q_linear_registry = LayerRegistry("QLinear")
q_conv1d_registry = LayerRegistry("QConv1d")
q_conv2d_registry = LayerRegistry("QConv2d")
q_conv3d_registry = LayerRegistry("QConv3d")


def all_layer_registries() -> List[LayerRegistry]:
    """
    Return all layer registries (one for each layer type: QLinear, QConv[1-3]d).

    Returns:
        A list of all layer registries.
    """
    return [
        q_conv1d_registry,
        q_conv2d_registry,
        q_conv3d_registry,
        q_linear_registry,
    ]


def convert_layers_to(
    new_mode: RuntimeMode,
    only: Optional[Iterable[Any]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> None:
    """
    Convert all wrapped layers (or a given subset of them) to a new mode.
    Args:
        new_mode: the new RuntimeMode
        only: optional white"list" (Iterable) of layers or wrapped layers which should be converted
        device: the new device for the layers
        verbose: whether to print which layers are being converted
    """
    for registry in all_layer_registries():
        registry.convert_layers_to(new_mode, only, device, verbose)


class QLinearImplementation(LayerImplementation):
    """Decorator for :class:`QLinear` implementations."""

    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_linear_registry, supports_modes)


class QConv1dImplementation(LayerImplementation):
    """Decorator for :class:`QConv1d` implementations."""

    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv1d_registry, supports_modes)


class QConv2dImplementation(LayerImplementation):
    """Decorator for :class:`QConv2d` implementations."""

    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv2d_registry, supports_modes)


class QConv3dImplementation(LayerImplementation):
    """Decorator for :class:`QConv3d` implementations."""

    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv3d_registry, supports_modes)
