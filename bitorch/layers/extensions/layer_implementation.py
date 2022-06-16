from abc import ABC
from typing import Any, TYPE_CHECKING

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
    def can_clone(cls, recipe: "LayerRecipe") -> bool:
        """
        Returns whether this layer class supports the implementation of a given layer recipe.

        Args:
            recipe (LayerRecipe): the layer which should be checked for cloning

        Returns:
            bool: Whether the layer can be cloned or not
        """
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe") -> Any:
        """
        Create a new layer based on a given layer recipe (can be expected to be from the default category).

        Args:
            recipe (LayerRecipe): the layer which should be cloned

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
    def can_clone(cls, recipe: "LayerRecipe") -> bool:
        return True

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe") -> Any:
        return cls(*recipe.args, **recipe.kwargs)


class CustomImplementationMixin(BaseImplementation, ABC):
    """Defines the class interface of a custom layer implementation of a certain layer type."""
    @classmethod
    def is_default_implementation(cls) -> bool:
        return False

    @classmethod
    def can_clone(cls, recipe: "LayerRecipe") -> bool:
        raise NotImplementedError("A custom layer should implement their own compatibility check.")

    @classmethod
    def create_clone_from(cls, recipe: "LayerRecipe") -> Any:
        raise NotImplementedError("A custom layer should implement a method to create a cloned layer.")
