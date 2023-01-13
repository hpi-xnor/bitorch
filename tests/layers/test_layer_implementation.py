import pickle
from typing import Any

import pytest

import bitorch
import torch
from bitorch import RuntimeMode
from bitorch.layers.extensions.layer_implementation import CustomImplementationMixin, DefaultImplementationMixin
from bitorch.layers.extensions import LayerRecipe, LayerImplementation, LayerRegistry
from bitorch.layers.extensions.layer_container import LayerContainer

TEST_MODE = RuntimeMode.INFERENCE_AUTO


class ExampleBase:
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def do_something(self):
        return f"{self.s}: {self.val} - made by {self.class_name()}"

    def class_name(self) -> str:
        return "BaseClass"


example_registry = LayerRegistry("Example")


class ExampleImplementation(LayerImplementation):
    def __init__(self, *args):
        super().__init__(example_registry, *args)


class ExampleComposed(DefaultImplementationMixin, ExampleBase):
    """Compose the default implementation"""

    pass


# create the decorated default implementation
Example = ExampleImplementation(RuntimeMode.DEFAULT)(ExampleComposed)


@ExampleImplementation(TEST_MODE)
class CustomLayerImplementation(CustomImplementationMixin, ExampleBase):
    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> bool:
        # assume this test class can only clone layers with 'vals' lower than 100
        val = recipe.kwargs.get("val", recipe.args[2] if 2 < len(recipe.args) else None)
        return val < 100, "val needs to be smaller than 100"

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe, device: torch.device) -> Any:
        return cls(recipe.layer.s, recipe.layer.val)

    def do_something(self):
        return f"{self.s}: {self.val} - made by {self.class_name()}"

    def class_name(self) -> str:
        return "CustomClass"


@pytest.fixture(scope="function", autouse=True)
def clean_environment():
    example_registry.clear()
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    example_registry.clear()
    bitorch.mode = RuntimeMode.DEFAULT


def test_recipe():
    s1 = Example("Hello World", val=21)
    s2 = Example("Hello World", 21)

    s1_recipe = example_registry.get_recipe_for(s1)
    assert s1_recipe.args[0] == "Hello World"
    assert s1_recipe.kwargs["val"] == 21

    s2_recipe = example_registry.get_recipe_for(s2)
    assert s2_recipe.args[0] == "Hello World"
    assert s2_recipe.args[1] == 21


def test_default_impl():
    print("bitorch test mode:", bitorch.mode)
    layer = Example("Hello World", val=21)
    assert layer.val == 21
    assert layer.class_name() == "BaseClass"
    assert isinstance(layer, Example.class_)
    assert isinstance(layer, LayerContainer)
    print(layer)
    # TODO: pickling is currently only possible in RAW mode
    # content = pickle.dumps(layer)

    # layer_loaded = pickle.loads(content)
    # assert layer_loaded.val == 21


def test_train_impl():
    bitorch.mode = TEST_MODE
    layer = Example("Hello World", val=21)
    assert layer.val == 21
    assert layer.class_name() == "CustomClass"
    assert isinstance(layer, CustomLayerImplementation)
    assert isinstance(layer, LayerContainer)


def test_raw_impl():
    bitorch.mode = RuntimeMode.RAW
    layer = Example("Hello World", val=21)
    assert layer.val == 21
    assert layer.class_name() == "BaseClass"
    assert isinstance(layer, Example.class_)
    assert not isinstance(layer, LayerContainer)

    content = pickle.dumps(layer)

    layer_loaded = pickle.loads(content)
    assert layer_loaded.val == 21


@pytest.mark.parametrize("val, is_supported", [(150, False), (50, True)])
def test_clone(val, is_supported):
    layer = Example("Hello World", val=val)
    recipe = example_registry.get_recipe_for(layer)
    if is_supported:
        replacement = example_registry.get_replacement(TEST_MODE, recipe)
        assert isinstance(replacement, CustomLayerImplementation)  # type: ignore
    else:
        with pytest.raises(RuntimeError) as e_info:
            _ = example_registry.get_replacement(TEST_MODE, recipe)
        error_message = str(e_info.value)
        assert e_info.typename == "RuntimeError"
        expected_key_strings = ["Example", "implementation", str(TEST_MODE), "val", "100"]
        for key in expected_key_strings:
            assert key in error_message
