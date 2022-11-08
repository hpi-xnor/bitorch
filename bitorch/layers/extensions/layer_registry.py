from typing import Set, Any, Optional, Iterable

import bitorch
import torch
from bitorch import RuntimeMode

from .layer_container import LayerContainer
from .layer_recipe import LayerRecipe
from .layer_registration import LayerImplementation


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

    def get_replacement(self, mode: RuntimeMode, recipe: LayerRecipe, device: Optional[torch.device] = None) -> Any:
        layer = self.get_layer(mode, recipe)
        return layer.get_replacement(recipe, device)

    def add_recipe(self, new_recipe: LayerRecipe) -> None:
        if self.is_replacing:
            return
        self._instance_recipes.add(new_recipe)

    def __contains__(self, item: Any) -> bool:
        return item.__class__ in map(lambda x: x.class_, self.layer_implementations)

    def register(self, layer: LayerImplementation) -> None:
        """
        Register a layer implementaiton in this registry.

        Args:
            layer: the layer to be registered
        """
        self.layer_implementations.add(layer)

    def get_layer(
        self, mode: Optional[RuntimeMode] = None, recipe: Optional[LayerRecipe] = None
    ) -> LayerImplementation:
        """
        Get a layer implementation compatible to the given mode and recipe.

        If no recipe is given, only compatibility with the mode is checked.
        If no mode is given, the current bitorch mode is used.

        Args:
            mode: mode that the layer implementation should support
            recipe: recipe that the layer implementation should be able to copy

        Returns:
            a LayerImplementation compatible with the given mode and recipe (if available)
        """
        if mode is None:
            mode = bitorch.mode
        available_layers = []
        unavailable_layers = []

        for implementation in self.layer_implementations:
            if not implementation.supports_mode(mode):
                continue
            if recipe:
                return_tuple = implementation.can_create_clone_from(recipe)
                if not isinstance(return_tuple, tuple) and len(return_tuple) == 2:
                    raise RuntimeError(f"{implementation.__class__} returned non-tuple on 'can_create_clone_from'.")
                can_be_used, message = return_tuple
                if not can_be_used:
                    unavailable_layers.append(f"    {implementation.__class__} unavailable because: {message}")
                    continue
            available_layers.append(implementation)

        if len(available_layers) > 1:
            RuntimeWarning(f"Multiple layer implementations available for '{self.name}' available (mode='{mode}').")
        if len(available_layers) == 0:
            base_error = f"No implementations for '{self.name}' available (mode='{mode}')."
            if len(unavailable_layers) > 0:
                raise RuntimeError("\n".join([base_error] + unavailable_layers))
            else:
                raise RuntimeError(base_error)
        return available_layers[0]

    def clear(self) -> None:
        while len(self._instance_recipes) > 0:
            self._instance_recipes.pop()

    def unregister_custom_implementations(self) -> None:
        to_remove = list(filter(lambda x: not x.is_default(), self.layer_implementations))
        for i in to_remove:
            self.layer_implementations.remove(i)

    def convert_layers_to(
        self,
        new_mode: RuntimeMode,
        only: Optional[Iterable[Any]] = None,
        device: Optional[torch.device] = None,
        verbose: bool = False,
    ) -> None:
        for recipe in list(self._instance_recipes):
            module = recipe.layer
            if only is not None and module.layer_implementation not in only and module not in only:
                continue
            assert isinstance(module, LayerContainer)
            if verbose:
                print("| Replacing layer in", module)
            replacement_module = self.get_replacement(new_mode, recipe, device)
            replacement_module.to(device)
            if verbose:
                print("- with:", replacement_module)
            module.replace_layer_implementation(replacement_module)
