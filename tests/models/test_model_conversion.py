from typing import Any

import pytest
from torch import nn
from torch.nn import functional as F

import bitorch
import bitorch.runtime_mode
from bitorch import RuntimeMode
from bitorch.datasets import MNIST
from bitorch.layers import QConv2d, QLinear
from bitorch.layers.extensions.layer_implementation import LayerRecipe, CustomImplementation
from bitorch.layers.qlinear import QLinearImplementation, q_linear_registry, QLinearBase
from bitorch.models import Model

TEST_MODE = RuntimeMode.INFERENCE_AUTO


class TestModel(Model):
    def __init__(self):
        super().__init__(dataset=MNIST)
        self.q_conv2d = QConv2d(1, 32, 3, 1, 1)
        self.q_linear = QLinear(784, 64)
        self._model = nn.Sequential(
            self.q_conv2d,
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.Flatten(),
            self.q_linear,
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self._model(x)
        output = F.log_softmax(x, dim=1)
        return output


@pytest.fixture
def get_decorated_test_impl():
    @QLinearImplementation(TEST_MODE)
    class QLinearTestImpl(CustomImplementation, nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            with bitorch.pause_wrapping():
                self._layer = QLinear(*args, **kwargs)
            self.is_test_implementation = True

        def forward(self, x):
            return self._layer(x)

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> bool:
            return True

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe) -> Any:
            return cls(*recipe.args, **recipe.kwargs)
    yield QLinearTestImpl
    q_linear_registry.clear()
    q_linear_registry.unregister_custom_implementations()


@pytest.fixture
def get_subclassed_test_impl():
    @QLinearImplementation(TEST_MODE)
    class QLinearTestImpl(CustomImplementation, QLinearBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_test_implementation = True

        @classmethod
        def can_clone(cls, recipe: LayerRecipe) -> bool:
            return True

        @classmethod
        def create_clone_from(cls, recipe: LayerRecipe) -> Any:
            return cls(*recipe.args, **recipe.kwargs)
    yield QLinearTestImpl
    q_linear_registry.clear()
    q_linear_registry.unregister_custom_implementations()


def test_convert_model_decorated(get_decorated_test_impl):
    net = TestModel()
    assert not hasattr(net.q_linear, "is_test_implementation")
    net.convert(TEST_MODE)
    assert hasattr(net.q_linear, "is_test_implementation") and net.q_linear.is_test_implementation


def test_convert_model_subclassed(get_subclassed_test_impl):
    net = TestModel()
    assert not hasattr(net.q_linear, "is_test_implementation")
    net.convert(TEST_MODE)
    assert hasattr(net.q_linear, "is_test_implementation") and net.q_linear.is_test_implementation
