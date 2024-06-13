from bitorch.models import ResnetE18
import torch
from torch.nn import Linear, Conv2d, ReLU
import pytest
import time

try:
    import torchvision

    torchvision_installed = True
except ModuleNotFoundError:
    torchvision_installed = False

TEST_DATA = [
    (ResnetE18, {"input_shape": (1, 3, 32, 32), "num_classes": 10}),
    (ResnetE18, {"input_shape": (1, 3, 224, 224), "num_classes": 1000}),
]

TEST_CUSTOM_DATA = [
    (ResnetE18, {"input_shape": (1, 6, 128, 128), "num_classes": 400}),
    (ResnetE18, {"input_shape": (1, 10, 320, 320), "num_classes": 33}),
]


@pytest.mark.skipif(not torchvision_installed, reason="torchvision is not installed")
@pytest.mark.parametrize("model, kwargs", TEST_DATA)
def test_from_pretrained(model, kwargs):
    m = model.from_pretrained(**kwargs)
    input_values = torch.randn(kwargs["input_shape"])

    result = m(input_values)
    assert result.shape == torch.Size([kwargs["input_shape"][0], kwargs["num_classes"]])


@pytest.mark.skipif(not torchvision_installed, reason="torchvision is not installed")
@pytest.mark.parametrize("model, kwargs", TEST_CUSTOM_DATA)
def test_from_pretrained_custom_shape(model, kwargs):
    with pytest.raises(RuntimeError):
        m = model.from_pretrained(**kwargs)


@pytest.mark.skipif(not torchvision_installed, reason="torchvision is not installed")
@pytest.mark.parametrize("model, kwargs", TEST_CUSTOM_DATA)
def test_adding_layers(model, kwargs):
    m1 = torch.nn.Sequential(
        Conv2d(kwargs["input_shape"][1], 37, kernel_size=3, stride=1, padding=1, bias=False),
        ReLU(),
        Conv2d(37, 43, kernel_size=3, stride=1, padding=1, bias=False),
    )

    m2 = torch.nn.Sequential(
        Linear(300, 100),
        ReLU(),
        Linear(100, kwargs["num_classes"]),
    )

    m = model.as_backbone(prepend_layers=m1, append_layers=m2, sanity_check=True)
    input_values = torch.randn(kwargs["input_shape"])

    result = m(input_values)
    assert result.shape == torch.Size([kwargs["input_shape"][0], kwargs["num_classes"]])


@pytest.mark.skipif(not torchvision_installed, reason="torchvision is not installed")
@pytest.mark.parametrize("model, kwargs", TEST_CUSTOM_DATA)
def test_changing_sizes(model, kwargs):
    m = model.as_backbone(input_size=kwargs["input_shape"][1:], output_size=[kwargs["num_classes"]], sanity_check=True)
    input_values = torch.randn(kwargs["input_shape"])

    result = m(input_values)
    assert result.shape == torch.Size([kwargs["input_shape"][0], kwargs["num_classes"]])
