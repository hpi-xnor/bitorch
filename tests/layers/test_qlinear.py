import pytest
from bitorch.layers.qlinear import QLinear
from torch.nn import Linear
from bitorch.layers.layerconfig import Quantizations
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
def test_qlinear(input_values):
    layer = QLinear(2, 2, bias=False, quantization="sign")
    full_precision_layer = Linear(2, 2, bias=False)
    assert isinstance(layer.quantize, type(Quantizations.sign()))

    test_weights = [[0.3, -1.4], [-0.3, 2.6]]

    layer.weight = Parameter(torch.tensor(test_weights))
    full_precision_layer.weight = Parameter(torch.tensor(test_weights))
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_hat = torch.tensor(input_values).float().requires_grad_(True)

    result = torch.tensor([x[0] - x[1], x[1] - x[0]])
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
