from typing import Any, Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import bitorch
import bitorch.runtime_mode
from bitorch import RuntimeMode
from bitorch.layers import QConv2d, QLinear
from bitorch.layers.extensions.layer_implementation import CustomImplementationMixin
from bitorch.layers.extensions import LayerRecipe
from bitorch.layers.qconv2d import QConv2dBase
from bitorch.layers.qlinear import QLinearBase
from bitorch.layers.register import (
    q_linear_registry,
    QLinearImplementation,
    q_conv2d_registry,
    QConv2dImplementation,
)
from bitorch.models import Model

TEST_MODE = RuntimeMode.INFERENCE_AUTO
MNIST = [(1, 1, 28, 28), 10, "MNIST"]


class _TestModel(Model):
    def __init__(self):
        super().__init__(input_shape=MNIST[0], num_classes=MNIST[1])
        self.q_conv2d = QConv2d(1, 32, 3, 1, 1)
        self.q_linear = QLinear(784, 64)
        self._model = nn.Sequential(
            self.q_conv2d,
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Flatten(),
            self.q_linear,
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self._model(x)
        output = F.log_softmax(x, dim=1)
        return output


def reset():
    bitorch.mode = RuntimeMode.DEFAULT
    for registry in (q_linear_registry, q_conv2d_registry):
        registry.unregister_custom_implementations()


@pytest.fixture
def get_decorated_impls():
    reset()

    @QLinearImplementation(TEST_MODE)
    class QLinearTestImpl(CustomImplementationMixin, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            with bitorch.pause_wrapping():
                self._layer = QLinear(*args, **kwargs)
            self.is_test_implementation = True

        def forward(self, x):
            return self._layer(x)

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> Tuple[bool, str]:
            return True, ""

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe, device: torch.device) -> Any:
            new_layer = cls(*recipe.args, **recipe.kwargs)
            new_layer._layer.weight = recipe.layer.weight
            new_layer._layer.bias = recipe.layer.bias
            return new_layer

    @QConv2dImplementation(TEST_MODE)
    class QConv2dTestImpl(CustomImplementationMixin, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            with bitorch.pause_wrapping():
                self._layer = QConv2d(*args, **kwargs)
            self.is_test_implementation = True

        def forward(self, x):
            return self._layer(x)

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> Tuple[bool, str]:
            return True, ""

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe, device: torch.device) -> Any:
            new_layer = cls(*recipe.args, **recipe.kwargs)
            new_layer._layer.weight = recipe.layer.weight
            new_layer._layer.bias = recipe.layer.bias
            return new_layer

    yield QLinearTestImpl, QConv2dTestImpl
    reset()


@pytest.fixture
def get_subclassed_impls():
    reset()

    @QLinearImplementation(TEST_MODE)
    class QLinearTestImpl(CustomImplementationMixin, QLinearBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_test_implementation = True

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> bool:
            return True, ""

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe, device: torch.device) -> Any:
            new_layer = cls(*recipe.args, **recipe.kwargs)
            new_layer.weight = recipe.layer.weight
            new_layer.bias = recipe.layer.bias
            return new_layer

    @QConv2dImplementation(TEST_MODE)
    class QConv2dTestImpl(CustomImplementationMixin, QConv2dBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_test_implementation = True

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> Tuple[bool, str]:
            return True, ""

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe, device: torch.device) -> Any:
            new_layer = cls(*recipe.args, **recipe.kwargs)
            new_layer.weight = recipe.layer.weight
            new_layer.bias = recipe.layer.bias
            return new_layer

    yield QLinearTestImpl, QConv2dTestImpl
    reset()


def _test():
    x = torch.rand(1, 1, 28, 28)
    net = _TestModel()

    assert not hasattr(net.q_linear, "is_test_implementation")
    assert not hasattr(net.q_conv2d, "is_test_implementation")
    y1 = net(x)

    net.convert(TEST_MODE)

    assert hasattr(net.q_linear, "is_test_implementation") and net.q_linear.is_test_implementation
    assert hasattr(net.q_conv2d, "is_test_implementation") and net.q_conv2d.is_test_implementation
    y2 = net(x)

    assert torch.equal(y1, y2)


def test_convert_model_decorated(get_decorated_impls):
    _test()


def test_convert_model_subclassed(get_subclassed_impls):
    _test()
