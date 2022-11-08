from abc import ABC
from typing import Optional, Any, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import LayerRecipe


class BaseImplementation:
    """Defines the class interface of a custom layer implementation of a certain layer type."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def is_default_implementation(cls) -> bool:
        """
        Returns:
            bool: whether this implementation is the default implementation of the current layer type
        """
        raise NotImplementedError("Should be implemented by subclass.")

    @classmethod
    def can_clone(cls, recipe: "LayerRecipe") -> Tuple[bool, str]:
        """
        Returns whether this layer class supports the implementation of a given layer recipe.

        Args:
            recipe (LayerRecipe): the layer which should be checked for cloning

        Returns:
            Whether the layer can be cloned or not and an info message if it can not be cloned
        """
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe", device: Optional[torch.device] = None) -> Any:
        """
        Create a new layer based on a given layer recipe (can be expected to be from the default category).

        Args:
            recipe: the layer which should be cloned
            device: the device on which the layer is going to be run

        Returns:
            A clone of the LayerRecipe in the current class implementation
        """
        raise NotImplementedError("A custom layer should implement a method to create a cloned layer.")


class DefaultImplementationMixin(BaseImplementation, ABC):
    """Defines the class interface of a default layer implementation of a certain layer type."""

    @classmethod
    def is_default_implementation(cls) -> bool:
        return True

    @classmethod
    def can_clone(cls, recipe: "LayerRecipe") -> Tuple[bool, str]:
        return True, ""

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe", device: Optional[torch.device] = None) -> Any:
        return cls(*recipe.args, **recipe.kwargs)


class CustomImplementationMixin(BaseImplementation, ABC):
    """Defines the class interface of a custom layer implementation of a certain layer type."""

    @classmethod
    def is_default_implementation(cls) -> bool:
        return False

    @classmethod
    def can_clone(cls, recipe: "LayerRecipe") -> Tuple[bool, str]:
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe", device: Optional[torch.device] = None) -> Any:
        raise NotImplementedError("A custom layer should implement a method to create a cloned layer.")
