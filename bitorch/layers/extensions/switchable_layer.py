from typing import Any


class LayerContainer:
    def __init__(self, impl_class: Any, *args: Any, **kwargs: Any) -> None:
        self._layer_implementation = impl_class(*args, **kwargs)

    def replace_layer_implementation(self, new_implementation: Any) -> None:
        self._layer_implementation = new_implementation

    def __getattr__(self, item: Any) -> Any:
        if item == "_layer_implementation":
            return self.__dict__[item]
        attr_value = getattr(self._layer_implementation, item)
        if attr_value == self._layer_implementation:
            return self
        if callable(attr_value):
            def patch_function(*args, **kwargs):
                fn_return_val = attr_value(*args, **kwargs)
                if fn_return_val == self._layer_implementation:
                    return self
                return fn_return_val
            return patch_function
        return attr_value

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._layer_implementation(*args, **kwargs)

    def __setattr__(self, key: Any, value: Any) -> None:
        if key == "_layer_implementation":
            self.__dict__[key] = value
            return
        setattr(self._layer_implementation, key, value)

    @property  # type: ignore[misc]
    def __class__(self) -> Any:
        return self._layer_implementation.__class__
