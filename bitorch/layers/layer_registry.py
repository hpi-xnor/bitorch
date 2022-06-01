from typing import Set, Optional, Any

import bitorch
from .. import RuntimeMode
from ..runtime_mode import runtime_mode_type


class LayerRegistry:
    def __init__(self, name: str) -> None:
        self.name = name
        self.registered_layers: Set[_LayerImplementation] = set()

    def register(self, layer: "_LayerImplementation") -> None:
        self.registered_layers.add(layer)

    def get_layer(self, mode: Optional[RuntimeMode] = None) -> "_LayerImplementation":
        if mode is None:
            mode = bitorch.mode
        available_layers = []
        for layer in self.registered_layers:
            if mode.is_supported_by(layer.supports_modes):
                available_layers.append(layer)
        if len(available_layers) > 1:
            RuntimeWarning(f"Multiple layer implementations available for '{self.name}' available (mode='{mode}').")
        if len(available_layers) == 0:
            raise RuntimeError(f"No layer implementation for '{self.name}' available (mode='{mode}').")
        return available_layers[0]


class _LayerImplementation:
    def __init__(self, registry: LayerRegistry, supports_modes: runtime_mode_type) -> None:
        self.registry = registry
        self.supports_modes = supports_modes
        assert self.supports_modes > 0, "Invalid mode given"
        self.__initialized = False
        self.class_: Any = None
        self.class_name = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.__initialized:
            self.__initialized = True
            self.class_ = args[0]
            self.class_name = self.class_.__name__
            self.registry.register(self)
            return self
        current_layer = self.registry.get_layer()
        if self == current_layer:
            # this class provides the correct implementation for the current mode (recursion stop)
            return self.class_(*args, **kwargs)
        # call this method again but on the correct base class
        return current_layer(*args, **kwargs)
