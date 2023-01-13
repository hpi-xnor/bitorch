import typing
from dataclasses import dataclass
from typing import TypeVar, Tuple, Any, Dict

if typing.TYPE_CHECKING:
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

    def get_positional_arg(self, pos: int) -> Any:
        """
        Get a positional argument from the stored args.

        Args:
            pos: the position of the argument if given as a positional arg

        Returns:
            the argument value retrieved
        """
        return self.args[pos]

    def get_arg(self, pos: int, key: str, default: T) -> T:
        """
        Get an argument from the stored args or kwargs.

        Args:
            pos: the position of the argument if given as a positional arg
            key: the name of the argument
            default: the default value of the argument

        Returns:
            the argument value retrieved
        """
        if len(self.args) > pos:
            return self.args[pos]
        return self.kwargs.get(key, default)
