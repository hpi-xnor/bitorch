from abc import ABC
from typing import Any, Union, Set, Optional

import bitorch
from bitorch import runtime_mode_type, RuntimeMode
from .switchable_layer import SwitchableLayer


class LayerImplementation(ABC):
    def __init__(self, registry: "LayerRegistry", supports_modes: runtime_mode_type) -> None:
        self.registry = registry
        assert RuntimeMode.is_combined_mode(supports_modes), f"invalid mode {supports_modes} given"
        self.supports_modes = supports_modes
        self.__initialized = False
        self.class_: Any = None
        self.class_name = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Union["LayerImplementation", SwitchableLayer]:
        if not self.__initialized:
            # this function is called once when @Decorator is used, we need to initialize this object correctly
            self.__initialized = True
            self.class_ = args[0]
            self.class_name = self.class_.__name__
            self.registry.register(self)
            return self

        # on later calls we need to provide the correct layer implementation
        correct_layer_implementation = self.registry.get_layer()
        if self == correct_layer_implementation:
            # this class provides the correct implementation for the current mode (recursion stop)
            return SwitchableLayer(self.class_, *args, **kwargs)
            # return self.class_(*args, **kwargs)
        # call this method again but on the correct base class
        return correct_layer_implementation(*args, **kwargs)


class LayerRegistry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._class = None
        self.registered_layers: Set[LayerImplementation] = set()

    def __contains__(self, item: Any) -> bool:
        return item.__class__ in map(lambda x: x.class_, self.registered_layers)

    def register(self, layer: LayerImplementation) -> None:
        self.registered_layers.add(layer)

    def get_layer(self, mode: Optional[RuntimeMode] = None) -> LayerImplementation:
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
