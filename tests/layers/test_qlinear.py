import pytest
import torch
from torch.nn import Parameter

from bitorch.layers.qactivation import QActivation
from bitorch.layers.qlinear import QLinear
from bitorch.quantizations import Sign, quantization_from_name

TEST_INPUT_DATA = [[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0], [0.3, -0.3], [1e12, -1e12]]


@pytest.mark.parametrize("input_values", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization", ["sign", Sign()])
def test_qlinear(input_values, quantization):
    layer = QLinear(2, 2, bias=False, weight_quantization=quantization, input_quantization=quantization)
    assert isinstance(layer.weight_quantization, quantization_from_name("sign"))
    assert isinstance(layer.input_quantization, quantization_from_name("sign"))

    test_weights = [[0.3, -1.4], [-0.3, 2.6]]

    input_activation = QActivation(quantization)

    layer.weight = Parameter(torch.tensor(test_weights))
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_hat = input_activation(torch.tensor(input_values).float().requires_grad_(True))

    result = torch.tensor([x_hat[0] - x_hat[1], x_hat[1] - x_hat[0]])
    y = layer(x)

    assert torch.equal(result, y)
