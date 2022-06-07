from abc import ABC
from dataclasses import dataclass
from typing import Any, Union, Set, Optional, Dict, Tuple

import bitorch
from bitorch import runtime_mode_type, RuntimeMode
from .switchable_layer import LayerContainer


@dataclass(eq=False, frozen=True)
class LayerRecipe:
    """Class to store args and kwargs used to create a particular layer. Allows to create other versions later on."""
    # registry: "LayerRegistry"
    container: "LayerContainer"
    args: Tuple[Any]
    kwargs: Dict[str, Any]


class LayerImplementation(ABC):
    """
    Superclass for storing different implementations of a common layer.

    It registers all decorated classes in the given registry and
    """
    def __init__(self, registry: "LayerRegistry", supports_modes: runtime_mode_type) -> None:
        self.registry = registry
        assert RuntimeMode.is_combined_mode(supports_modes), f"invalid mode {supports_modes} given"
        self.supports_modes = supports_modes
        self.__initialized = False
        self.class_: Any = None
        self.class_name = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Union["LayerImplementation", LayerContainer]:
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
            if self.registry.is_replacing:
                return self.class_(*args, **kwargs)
            else:
                layer_container = LayerContainer(self.class_, *args, **kwargs)
                self.registry.add_recipe(LayerRecipe(container=layer_container, args=args, kwargs=kwargs))
                return layer_container
        # call this method again but on the correct base class
        return correct_layer_implementation(*args, **kwargs)


class LayerRegistry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._class = None
        self.registered_layers: Set[LayerImplementation] = set()
        self.instance_recipes: Set[LayerRecipe] = set()
        self.is_replacing = False

    def get_replacement(self, *args: Any, **kwargs: Any) -> Any:
        self.is_replacing = True
        replacement_layer = self.get_layer()(*args, **kwargs)
        self.is_replacing = False
        return replacement_layer

    def add_recipe(self, new_recipe: LayerRecipe) -> None:
        if self.is_replacing:
            return
        self.instance_recipes.add(new_recipe)

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
