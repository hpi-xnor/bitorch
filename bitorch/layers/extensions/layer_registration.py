from abc import ABC
from typing import Optional, Any, Type, Union, Tuple, TYPE_CHECKING

import torch

import bitorch
from bitorch import runtime_mode_type, RuntimeMode
from .layer_container import LayerContainer
from .layer_implementation import DefaultImplementationMixin, BaseImplementation, CustomImplementationMixin
from .layer_recipe import LayerRecipe

if TYPE_CHECKING:
    from .layer_registry import LayerRegistry


class LayerImplementation(ABC):
    """
    Superclass for storing different implementations of a common layer type.

    It registers all decorated classes in the given registry. On creation of a decorated class, it
    wraps the created class object in a layer container and stores the arguments used to create the layer.
    It also captures which RuntimeMode(s) is/are supported by an implementation.
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

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> Union["LayerImplementation", Type[BaseImplementation], LayerContainer]:
        if not self.__initialized:
            # this object is called once when @Decorator is used, we need to initialize
            return self._initialize(*args, **kwargs)

        if bitorch.mode == RuntimeMode.RAW:
            return self.class_(*args, **kwargs)  # type: ignore

        # on later calls we need to provide the correct layer implementation
        return self._provide_layer_implementation(*args, **kwargs)

    def _initialize(self, class_: Type[BaseImplementation]) -> Union["LayerImplementation", Type[BaseImplementation]]:
        self.__initialized = True
        self.class_ = class_
        self.class_name = self.class_.__name__
        self.registry.register(self)
        if self._supported_modes == RuntimeMode.DEFAULT:
            assert issubclass(
                self.class_, DefaultImplementationMixin
            ), f"{self.class_name} should be a subclass of DefaultLayerImplementation."
            # provide this wrapper
            return self
        else:
            assert issubclass(self.class_, CustomImplementationMixin), (
                f"{self.class_name} should be a subclass of CustomImplementationInterface (and it should "
                f"implement the corresponding class methods)."
            )
            # after we have registered custom implementations, we do not interfere anymore
            return self.class_

    def _provide_layer_implementation(self, *args: Any, **kwargs: Any) -> LayerContainer:
        correct_layer_implementation = self.registry.get_layer()
        if self == correct_layer_implementation:
            # this class provides the correct implementation for the current mode (recursion stop)
            layer_container = LayerContainer(self.class_, *args, **kwargs)
            self.registry.add_recipe(layer_container.recipe)
            return layer_container
        # call this method again but on the correct base class
        return correct_layer_implementation._provide_layer_implementation(*args, **kwargs)

    def supports_mode(self, mode: RuntimeMode) -> bool:
        """
        Check whether this layer implementation supports a given RuntimeMode.
        Args:
            mode: the runtime mode that should be supported

        Returns:
            True if the given mode is supported, False otherwise
        """
        return mode.is_supported_by(self._supported_modes)

    def can_create_clone_from(self, recipe: LayerRecipe) -> Tuple[bool, str]:
        return self.class_.can_clone(recipe)

    def get_replacement(self, recipe: LayerRecipe, device: Optional[torch.device] = None) -> Any:
        return self.class_.create_clone_from(recipe, device)

    def is_default(self) -> bool:
        return self.class_.is_default_implementation()
