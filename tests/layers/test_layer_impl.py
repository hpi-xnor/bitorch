from typing import Any

import pytest

import bitorch
from bitorch import RuntimeMode
from bitorch.layers.extensions.layer_implementation import LayerImplementation, LayerRegistry, \
    CustomImplementation, DefaultImplementation, LayerRecipe
from bitorch.layers.extensions.switchable_layer import LayerContainer

TEST_MODE = RuntimeMode.INFERENCE_AUTO

test_registry = LayerRegistry("TestLayer")


class TestLayerImplementation(LayerImplementation):
    def __init__(self, *args):
        super().__init__(test_registry, *args)


@TestLayerImplementation(RuntimeMode.DEFAULT)
class TestLayerDefaultMode(DefaultImplementation):
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def do_something(self):
        return f"{self.s}: {self.val} - made by {self.class_name()}"

    def class_name(self) -> str:
        return self.__class__.__name__


@TestLayerImplementation(TEST_MODE)
class TestLayerCustomMode(CustomImplementation):
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    @classmethod
    def can_clone(cls, recipe: LayerRecipe) -> bool:
        # assume this test class can only clone layers with 'vals' lower than 100
        val = recipe.kwargs.get("val", recipe.args[2] if 2 < len(recipe.args) else None)
        return val < 100

    @classmethod
    def create_clone_from(cls, recipe: LayerRecipe) -> Any:
        return cls(recipe.layer.s, recipe.layer.val)

    def do_something(self):
        return f"{self.s}: {self.val} - made by {self.class_name()}"

    def class_name(self) -> str:
        return self.__class__.__name__


TestLayer = TestLayerDefaultMode


@pytest.fixture(scope='function', autouse=True)
def clean_environment():
    test_registry.clear()
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    test_registry.clear()
    bitorch.mode = RuntimeMode.DEFAULT


def test_recipe():
    s1 = TestLayer("Hello World", val=21)
    s2 = TestLayer("Hello World", 21)

    s1_recipe = test_registry.get_recipe_for(s1)
    assert s1_recipe.args[0] == "Hello World"
    assert s1_recipe.kwargs["val"] == 21

    s2_recipe = test_registry.get_recipe_for(s2)
    assert s2_recipe.args[0] == "Hello World"
    assert s2_recipe.args[1] == 21


def test_default_impl():
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerDefaultMode"
    assert isinstance(s, TestLayerDefaultMode.class_)
    assert isinstance(s, LayerContainer)


def test_train_impl():
    bitorch.mode = TEST_MODE
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerCustomMode"
    assert isinstance(s, TestLayerCustomMode.class_)
    assert isinstance(s, LayerContainer)


def test_raw_impl():
    bitorch.mode = RuntimeMode.RAW
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerDefaultMode"
    assert isinstance(s, TestLayer.class_)
    assert not isinstance(s, LayerContainer)


@pytest.mark.parametrize("val, is_supported", [(150, False), (50, True)])
def test_clone(val, is_supported):
    s = TestLayer("Hello World", val=val)
    s_recipe = test_registry.get_recipe_for(s)
    if is_supported:
        replacement = test_registry.get_replacement(TEST_MODE, s_recipe)
        assert isinstance(replacement, TestLayerCustomMode.class_)  # type: ignore
    else:
        with pytest.raises(RuntimeError) as e_info:
            _ = test_registry.get_replacement(TEST_MODE, s_recipe)
        error_message = str(e_info.value)
        assert e_info.typename == "RuntimeError"
        expected_key_strings = ["TestLayer", "layer implementation", str(TEST_MODE)]
        assert all(key in error_message for key in expected_key_strings)
