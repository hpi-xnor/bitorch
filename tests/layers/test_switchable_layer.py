import pytest
import torch
from torch import nn

from bitorch.layers.extensions.switchable_layer import LayerContainer


class Layer(nn.Module):
    def __init__(self, x=10):
        super().__init__()
        self.x = x

    def foo(self):
        assert isinstance(self.x, int)
        return "foo"

    @property
    def self_property(self):
        return self

    def self_function(self):
        return self


@pytest.mark.parametrize("test_wrapped_layer", [False, True])
def test_switchable_layer(test_wrapped_layer):
    if test_wrapped_layer:
        layer = LayerContainer(Layer, 42)
    else:
        layer = Layer(42)
    assert layer.x == 42
    layer.x = 3
    assert layer.x == 3
    assert layer.self_function() == layer
    assert layer.self_property == layer

    def test_class_assertions(layer_):
        assert isinstance(layer_, nn.Module)
        assert isinstance(layer_, Layer)
        assert test_wrapped_layer == isinstance(layer_, LayerContainer)

    test_class_assertions(layer)

    moved_layer = layer.to(torch.device("cpu"))

    test_class_assertions(moved_layer)
    assert layer == moved_layer
