from typing import Set, Any, Optional, Iterable

import torch

import bitorch
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
            raise RuntimeError(f"No implementation for '{self.name}' available (mode='{mode}').")
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
            only: Optional[Iterable[Any]] = None,
            device: torch.device = None,
            verbose: bool = False
    ) -> None:
        for recipe in list(self._instance_recipes):
            module = recipe.layer
            if only is not None and module.layer_implementation not in only and module not in only:
                continue
            assert isinstance(module, LayerContainer)
            if verbose:
                print("Converting", module)
            replacement_module = self.get_replacement(new_mode, recipe)
            replacement_module.to(device)
            module.replace_layer_implementation(replacement_module)
