from abc import ABC
from dataclasses import dataclass
from typing import Any, Union, Set, Optional, Dict, Tuple, Type, Iterable

import torch
from torch import nn

import bitorch
from bitorch import runtime_mode_type, RuntimeMode
from .switchable_layer import LayerContainer


@dataclass(eq=False, frozen=True)
class LayerRecipe:
    """
    Data class to store a layer object and the arguments used to create it.
    It allows to create other implementations of the same layer later on.
    """
    layer: "LayerContainer"
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]


class BaseImplementation:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    """Defines the class interface of a custom layer implementation of a certain layer type."""
    @classmethod
    def is_default_implementation(cls) -> bool:
        """
        Returns:
            bool: whether this implementation is the default implementation of the current layer type
        """
        raise NotImplementedError("Should be implemented by subclass.")

    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> bool:
        """
        Returns whether this layer class supports the implementation of a given layer recipe.

        Args:
            recipe (LayerRecipe): the layer which should be checked for cloning

        Returns:
            bool: Whether the layer can be cloned or not
        """
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe) -> Any:
        """
        Create a new layer based on a given layer recipe (can be expected to be from the default category).

        Args:
            recipe (LayerRecipe): the layer which should be cloned

        Returns:
            A clone of the LayerRecipe in the current class implementation
        """
        raise NotImplementedError("A custom layer should implement a method to create a cloned layer.")


class DefaultImplementation(BaseImplementation, ABC):
    """Defines the class interface of a default layer implementation of a certain layer type."""
    @classmethod
    def is_default_implementation(cls) -> bool:
        return True

    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> bool:
        return True

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe) -> Any:
        return cls(*recipe.args, **recipe.kwargs)


class CustomImplementation(BaseImplementation, ABC):
    """Defines the class interface of a custom layer implementation of a certain layer type."""
    @classmethod
    def is_default_implementation(cls) -> bool:
        return False

    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> bool:
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe) -> Any:
        raise NotImplementedError("A custom layer should implement a method to create a cloned layer.")


class LayerImplementation(ABC):
    """
    Superclass for storing different implementations of a common layer type.

    It registers all decorated classes in the given registry. On creation of a decorated class, it
    wraps the created class object in a layer container and stores the arguments used to create the layer.
    """

    registry: "LayerRegistry"
    class_: Type[BaseImplementation]
    class_name: str
    _supported_modes: runtime_mode_type
    __initialized: bool

    def __init__(self, registry: "LayerRegistry", supported_modes: runtime_mode_type) -> None:
        """
        Define an implementation decorator for a certain type of layer. All implementations and objects of this type of
        layer are stored in the given registry.

        Args:
            registry: the registry which should store the implementation and objects of this layer type
            supported_modes: the mode supported by the registering implementation
        """
        self.registry = registry
        assert RuntimeMode.is_combined_mode(supported_modes), f"invalid mode {supported_modes} given"
        self._supported_modes = supported_modes
        self.__initialized = False
        self.class_ = None  # type: ignore
        self.class_name = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Union["LayerImplementation", LayerContainer, nn.Module]:
        if not self.__initialized:
            # this object is called once when @Decorator is used, we need to initialize
            return self._initialize(*args, **kwargs)

        if bitorch.mode == RuntimeMode.RAW:
            return self.class_(*args, **kwargs)  # type: ignore

        # on later calls we need to provide the correct layer implementation
        return self._provide_layer_implementation(*args, **kwargs)

    def _initialize(self, class_: Type[BaseImplementation]) -> "LayerImplementation":
        self.__initialized = True
        self.class_ = class_
        self.class_name = self.class_.__name__
        if self._supported_modes == RuntimeMode.DEFAULT:
            assert issubclass(self.class_, DefaultImplementation), \
                f"{self.class_name} should be a subclass of DefaultLayerImplementation."
        else:
            assert issubclass(self.class_, CustomImplementation), \
                f"{self.class_name} should be a subclass of CustomImplementationInterface (and it should " \
                f"implement the corresponding class methods)."
        self.registry.register(self)
        return self

    def _provide_layer_implementation(self, *args: Any, **kwargs: Any) -> LayerContainer:
        correct_layer_implementation = self.registry.get_layer()
        if self == correct_layer_implementation:
            # this class provides the correct implementation for the current mode (recursion stop)
            layer_container = LayerContainer(self.class_, *args, **kwargs)
            self.registry.add_recipe(LayerRecipe(layer=layer_container, args=args, kwargs=kwargs))
            return layer_container
        # call this method again but on the correct base class
        return correct_layer_implementation._provide_layer_implementation(*args, **kwargs)

    def supports_mode(self, mode: RuntimeMode) -> bool:
        return mode.is_supported_by(self._supported_modes)

    def can_create_clone_from(self, recipe: LayerRecipe) -> bool:
        return self.class_.can_clone(recipe)

    def get_replacement(self, recipe: LayerRecipe) -> Any:
        return self.class_.create_clone_from(recipe)

    def is_default(self) -> bool:
        return self.class_.is_default_implementation()


class LayerRegistry:
    """
    Stores all available implementations (and their supported modes) for a certain type of layer.
    It also wraps these implementations and stores references to them, so they can be replaced easily.
    Needs to be subclassed for each type of layer.
    """
    def __init__(self, name: str) -> None:
        self.name = name
        self._class = None
        self.layer_implementations: Set[LayerImplementation] = set()
        self._instance_recipes: Set[LayerRecipe] = set()
        self.is_replacing = False

    @property
    def layer_instances(self) -> Set["LayerContainer"]:
        return set(x.layer for x in self._instance_recipes)

    def get_recipe_for(self, layer: Any) -> Optional["LayerRecipe"]:
        if layer not in map(lambda x: x.layer, self._instance_recipes):
            return None
        return next(filter(lambda x: x.layer == layer, self._instance_recipes))

    def get_replacement(self, mode: RuntimeMode, recipe: LayerRecipe) -> Any:
        layer = self.get_layer(mode, recipe)
        return layer.get_replacement(recipe)

    def add_recipe(self, new_recipe: LayerRecipe) -> None:
        if self.is_replacing:
            return
        self._instance_recipes.add(new_recipe)

    def __contains__(self, item: Any) -> bool:
        return item.__class__ in map(lambda x: x.class_, self.layer_implementations)

    def register(self, layer: LayerImplementation) -> None:
        self.layer_implementations.add(layer)

    def get_layer(
            self, mode: Optional[RuntimeMode] = None, recipe: Optional[LayerRecipe] = None
    ) -> LayerImplementation:
        if mode is None:
            mode = bitorch.mode
        available_layers = []
        for implementation in self.layer_implementations:
            if not implementation.supports_mode(mode):
                continue
            if recipe and not implementation.can_create_clone_from(recipe):
                continue
            available_layers.append(implementation)
        if len(available_layers) > 1:
            RuntimeWarning(f"Multiple layer implementations available for '{self.name}' available (mode='{mode}').")
        if len(available_layers) == 0:
            raise RuntimeError(f"No layer implementation for '{self.name}' available (mode='{mode}').")
        return available_layers[0]

    def clear(self) -> None:
        while len(self._instance_recipes) > 0:
            self._instance_recipes.pop()

    def unregister_custom_implementations(self) -> None:
        to_remove = list(filter(lambda x: not x.is_default(), self.layer_implementations))
        for i in to_remove:
            self.layer_implementations.remove(i)

    def convert_layers_to(
            self, new_mode: RuntimeMode,
            filter_: Optional[Iterable[Any]] = None,
            device: torch.device = None,
            verbose: bool = False
    ) -> None:
        for recipe in self._instance_recipes:
            module = recipe.layer
            if filter_ is not None and module.layer_implementation not in filter_:
                continue
            assert isinstance(module, LayerContainer)
            if verbose:
                print("Converting", module)
            replacement_module = self.get_replacement(new_mode, recipe)
            replacement_module.to(device)
            module.replace_layer_implementation(replacement_module)
