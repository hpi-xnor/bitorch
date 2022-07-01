import pytest
import torch
from bitorch.layers import Pact

TEST_THRESHOLDS = [0.5, 1.0, 2.0]


@pytest.mark.parametrize("alpha", TEST_THRESHOLDS)
def test_qactivation(alpha):
    pact = Pact(bits=2)

    x = torch.Tensor(torch.rand(100)).float().requires_grad_(True)
    pact.alpha = torch.nn.parameter.Parameter(torch.tensor(alpha))

    y = pact(x)

    clamped = torch.clamp(x, min=0, max=alpha)
    scale = 3.0 / alpha
    quantized = torch.round(clamped * scale) / scale

    assert torch.equal(quantized, y)

    y.backward(x)
    expected_gradient = torch.where((x >= 0) & (x <= alpha), x, torch.tensor(0.0))
    assert torch.equal(expected_gradient, x.grad)
