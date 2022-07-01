from typing import Any, TypeVar, Type, Generic

from bitorch.layers.extensions.layer_recipe import LayerRecipe

T = TypeVar("T")


class LayerContainer(Generic[T]):
    """This class wraps another layer - but the internally contained class can be swapped out during runtime."""

    internal_variable_names = [
        "_layer_implementation",
        "_recipe",
    ]

    patch = [
        "to",
    ]

    def __init__(self, impl_class: Type[T], *args: Any, **kwargs: Any) -> None:
        """
        Wrap a new object based on the given class, positional arguments, and keyword arguments.
        Args:
            impl_class: class of the new object
            *args: positional arguments of the new object
            **kwargs: keyword arguments of the new object
        """
        self._layer_implementation = impl_class(*args, **kwargs)
        self._recipe = LayerRecipe(layer=self, args=args, kwargs=kwargs)

    def replace_layer_implementation(self, new_implementation: T) -> None:
        """
        Replace the internally stored layer object with the given one.
        Args:
            new_implementation: new class which should replace the previous implementation.
        """
        self._layer_implementation = new_implementation

    def __getattr__(self, item: Any) -> Any:
        if item in self.internal_variable_names:
            return self.__dict__[item]
        attr_value = getattr(self._layer_implementation, item)
        if attr_value == self._layer_implementation:
            return self
        if callable(attr_value) and item in self.patch:
            # patch return values of all functions/classes defined in self.patch
            # they should return this LayerContainer instead of themselves
            # required for e.g. pytorch's .to(device) function
            other = self

            class Patch:
                def __call__(self, *args: Any, **kwargs: Any) -> Any:
                    fn_return_val = attr_value(*args, **kwargs)
                    if fn_return_val == other._layer_implementation:
                        return other
                    return fn_return_val

                def __getattr__(self, item_: Any) -> Any:
                    return getattr(attr_value, item_)

                # needed for tests:
                @property  # type: ignore[misc]
                def __class__(self) -> Any:
                    return attr_value.__class__

            return Patch()
        return attr_value

    def __repr__(self) -> "str":
        return f"LayerContainer (at {hex(id(self))}), contains: {self._layer_implementation}"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._layer_implementation(*args, **kwargs)  # type:ignore[operator]

    def __setattr__(self, key: Any, value: Any) -> None:
        if key in self.internal_variable_names:
            self.__dict__[key] = value
            return
        setattr(self._layer_implementation, key, value)

    @property  # type: ignore[misc]
    def __class__(self) -> Type[T]:  # type: ignore
        return self._layer_implementation.__class__

    @property
    def layer_implementation(self) -> T:
        """
        Access the internally wrapped layer object directly.
        Returns:
            the internal layer object
        """
        return self._layer_implementation

    @property
    def recipe(self) -> LayerRecipe:
        return self._recipe
