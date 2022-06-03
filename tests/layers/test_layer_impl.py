import pytest

import bitorch
from bitorch import RuntimeMode
from bitorch.layers.extensions.layer_implementation import LayerImplementation, LayerRegistry

_registry = LayerRegistry("TestLayer")


class TestLayerImplementation(LayerImplementation):
    def __init__(self, *args):
        super().__init__(_registry, *args)


@TestLayerImplementation(RuntimeMode.DEFAULT)
class TestLayerDefaultMode:
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def class_name(self) -> str:
        return self.__class__.__name__


@TestLayerImplementation(RuntimeMode.INFERENCE_AUTO)
class TestLayerOtherMode:
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def class_name(self) -> str:
        return self.__class__.__name__


TestLayer = TestLayerDefaultMode


@pytest.fixture(scope='function', autouse=True)
def set_default_mode():
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_default_impl():
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerDefaultMode"


def test_train_impl():
    bitorch.mode = RuntimeMode.INFERENCE_AUTO
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerOtherMode"

