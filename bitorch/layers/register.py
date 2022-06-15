from typing import List

from bitorch import runtime_mode_type
from bitorch.layers.extensions import LayerImplementation, LayerRegistry

q_linear_registry = LayerRegistry("QLinear")
q_conv1d_registry = LayerRegistry("QConv1d")
q_conv2d_registry = LayerRegistry("QConv2d")
q_conv3d_registry = LayerRegistry("QConv3d")


def all_layer_registries() -> List[LayerRegistry]:
    return [
        q_conv1d_registry,
        q_conv2d_registry,
        q_conv3d_registry,
        q_linear_registry,
    ]


class QLinearImplementation(LayerImplementation):
    """
    Decorator for :class:`QLinear` implementations, captures which RuntimeMode(s) is/are supported by an implementation.
    """
    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_linear_registry, supports_modes)


class QConv1dImplementation(LayerImplementation):
    """
    Decorator for :class:`QConv1d` implementations, captures which RuntimeMode(s) is/are supported by an implementation.
    """
    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv1d_registry, supports_modes)


class QConv2dImplementation(LayerImplementation):
    """
    Decorator for :class:`QConv2d` implementations, captures which RuntimeMode(s) is/are supported by an implementation.
    """
    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv2d_registry, supports_modes)


class QConv3dImplementation(LayerImplementation):
    """
    Decorator for :class:`QConv3d` implementations, captures which RuntimeMode(s) is/are supported by an implementation.
    """
    def __init__(self, supports_modes: runtime_mode_type) -> None:
        """
        Args:
            supports_modes:  RuntimeMode(s) that is/are supported by an implementation
        """
        super().__init__(q_conv3d_registry, supports_modes)
