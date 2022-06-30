from typing import Dict, Any

from .extensions import LayerRecipe


class QConvArgsProviderMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_args_as_kwargs(recipe: LayerRecipe) -> Dict[str, Any]:
        """
        Gather all arguments that were used to create a QLinear layer with argument names.
        Can be used to recreate a layer with identical arguments.

        Returns:
            A dictionary with all arguments (key is the argument name as a string even for positional arguments)
        """
        return {
            "in_channels": recipe.get_positional_arg(0),
            "out_channels": recipe.get_positional_arg(1),
            "kernel_size": recipe.get_positional_arg(2),
            "stride": recipe.get_arg(3, "stride", None),
            "padding": recipe.get_arg(4, "padding", None),
            "dilation": recipe.get_arg(5, "dilation", None),
            "groups": recipe.get_arg(6, "groups", None),
            "bias": recipe.get_arg(7, "bias", True),
            "padding_mode": recipe.get_arg(8, "padding_mode", None),
            "device": recipe.get_arg(9, "device", None),
            "dtype": recipe.get_arg(10, "dtype", None),
        }
