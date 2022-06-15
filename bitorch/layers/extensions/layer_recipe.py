from dataclasses import dataclass
from typing import TypeVar, Tuple, Any, Dict

from .layer_container import LayerContainer

T = TypeVar("T")


@dataclass(eq=False, frozen=True)
class LayerRecipe:
    """
    Data class to store a layer object and the arguments used to create it.
    It allows to create other implementations of the same layer later on.
    """
    layer: "LayerContainer"
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]

    def get_by_position_or_key(self, pos: int, key: str, default: T) -> T:
        if len(self.args) > pos:
            return self.args[pos]
        return self.kwargs.get(key, default)
