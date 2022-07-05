import pickle

import pytest
import torch
from torch import nn

from bitorch.layers.extensions.layer_container import LayerContainer


class Foo:
    pass


class Layer(nn.Module):
    def __init__(self, x=10):
        super().__init__()
        self.x = x
        self.foo = Foo()

    def get_foo(self):
        return self.foo

    @property
    def self_property(self):
        return self

    def self_function(self):
        return self


class _LayerContainer(LayerContainer):
    patch = LayerContainer.patch + [
        "self_function",
    ]


@pytest.mark.parametrize("test_wrapped_layer", [False, True])
def test_switchable_layer(test_wrapped_layer):
    if test_wrapped_layer:
        layer = _LayerContainer(Layer, 42)
    else:
        layer = Layer(42)
    assert layer.x == 42
    layer.x = 3
    assert layer.x == 3
    assert layer.self_function() == layer
    assert layer.self_property == layer

    assert isinstance(layer, nn.Module)
    assert isinstance(layer, Layer)
    assert isinstance(layer.foo, Foo)
    assert isinstance(layer.get_foo(), Foo)
    assert test_wrapped_layer == isinstance(layer, LayerContainer)

    moved_layer = layer.to(torch.device("cpu"))

    assert isinstance(layer, nn.Module)
    assert isinstance(layer, Layer)
    assert isinstance(layer.foo, Foo)
    assert isinstance(layer.get_foo(), Foo)
    assert test_wrapped_layer == isinstance(layer, LayerContainer)

    assert layer == moved_layer
