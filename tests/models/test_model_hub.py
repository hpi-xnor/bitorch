from bitorch.models import ResnetE18
import torch
import pytest
import time

TEST_DATA = [
    (ResnetE18, {"input_shape": (1, 3, 32, 32), "num_classes": 10}),
]


@pytest.mark.parametrize("model, kwargs", TEST_DATA)
def test_model_hub(model, kwargs):
    m = model.from_pretrained(**kwargs)
    input_values = torch.randn(kwargs["input_shape"])

    result = m(input_values)
    assert result.shape == torch.Size([kwargs["input_shape"][0], kwargs["num_classes"]])
