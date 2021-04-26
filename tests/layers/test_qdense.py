import pytest
from bitorch.layers.qlinear import QLinear
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
    assert isinstance(layer.quantize, type(Quantizations.sign()))

    layer.weight = Parameter(torch.tensor([[0.3, -0.3], [-0.3, 0.3]]))
    x = torch.tensor(input_values).float().requires_grad_(True)

    result = torch.tensor([x[0] - x[1], x[1] - x[0]])
    y = layer(x)

    assert torch.equal(result, y)
    y.backward(x)

    computed_gradient = x.grad.clone()
    assert torch.equal(computed_gradient, y)
