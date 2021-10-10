import pytest
from bitorch.layers.qlinear import QLinear
from bitorch.layers.qactivation import QActivation
from torch.nn import Linear
from bitorch.quantizations import Sign, quantization_from_name
import torch
from torch.nn import Parameter


TEST_INPUT_DATA = [
    [0., 0.],
    [1., 0.],
    [-1., 1.],
    [0.3, -0.3],
    [1e12, -1e12]
]


@pytest.mark.parametrize("input_values", TEST_INPUT_DATA)
@pytest.mark.parametrize("quantization", ["sign", Sign()])
def test_qlinear(input_values, quantization):
    layer = QLinear(2, 2, bias=False, weight_quantization="sign", input_quantization="sign")
    full_precision_layer = Linear(2, 2, bias=False)
    assert isinstance(layer.weight_quantize, quantization_from_name("sign"))

    test_weights = [[0.3, -1.4], [-0.3, 2.6]]

    input_activation = QActivation(quantization)

    layer.weight = Parameter(torch.tensor(test_weights))
    full_precision_layer.weight = Parameter(torch.tensor(test_weights))
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_hat = input_activation(torch.tensor(input_values).float().requires_grad_(True))

    result = torch.tensor([x_hat[0] - x_hat[1], x_hat[1] - x_hat[0]])
    y = layer(x)

    assert torch.equal(result, y)
    y.backward(x)

    computed_gradient = x.grad.clone()
    assert torch.equal(computed_gradient, y)

    y_hat = full_precision_layer(x_hat)
    y_hat.backward(x_hat)

    # now assert that weight gradient got either canceled correctly or where passed through is equal to weight gradient
    # of full precision layer
    full_precision_grad = full_precision_layer.weight.grad.clone()
    correct_gradient = torch.tensor([[full_precision_grad[0][0], 0.], [full_precision_grad[1][0], 0.]])
    assert torch.equal(correct_gradient, layer.weight.grad)
