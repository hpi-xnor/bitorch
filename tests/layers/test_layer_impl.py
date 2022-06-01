from typing import Tuple

import pytest

import bitorch
from bitorch import RuntimeMode
from bitorch.layers.layer_registry import LayerRegistry, _LayerImplementation

_registry = LayerRegistry("TestLayer")


class TestLayerImplementation(_LayerImplementation):
    def __init__(self, *args):
        super().__init__(_registry, *args)


@TestLayerImplementation(RuntimeMode.DEFAULT)
class TestLayerDefault:
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def class_name(self) -> str:
        return self.__class__.__name__


@TestLayerImplementation(RuntimeMode.TRAIN)
class TestLayerTrain:
    def __init__(self, s: str, val: int = 42) -> None:
        self.s = s
        self.val = val

    def class_name(self) -> str:
        return self.__class__.__name__


TestLayer = TestLayerDefault


@pytest.fixture(scope='function', autouse=True)
def set_default_mode():
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_default_impl():
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerDefault"


def test_train_impl():
    bitorch.mode = RuntimeMode.TRAIN
    s = TestLayer("Hello World", val=21)
    assert s.val == 21
    assert s.class_name() == "TestLayerTrain"
