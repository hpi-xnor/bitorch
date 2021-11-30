import torch
import pytest
from bitorch.quantizations import (
    Quantization,
    InputDoReFa,
    WeightDoReFa,
    Sign,
    ApproxSign,
    SteHeaviside,
    SwishSign,
    quantization_from_name
)

TEST_INPUT_DATA = [
    (Sign(), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5], [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (ApproxSign(), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5], [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.4, 2.0, 1.4, 0.0, 0.0]),
    (SteHeaviside(), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5],
     [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (SwishSign(5.0), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5], [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
     [-0.03, -0.195, 1.562, 5.0, 1.562, -0.195, -0.03]),
    (InputDoReFa(bits=2), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5], [0.0, 0.0, 0.0, 0.0, 1.0 / 3.0, 1.0, 1.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WeightDoReFa(bits=2), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5],
     [-1.0, -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    (InputDoReFa(bits=1), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    (WeightDoReFa(bits=1), [-1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5],
     [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]


@pytest.mark.parametrize("quantization, input_values, expected_output, expected_gradient_factors", TEST_INPUT_DATA)
def test_quantizations(
        quantization: Quantization,
        input_values: list,
        expected_output: list,
        expected_gradient_factors: list) -> None:
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_exp = torch.tensor(expected_output).float().requires_grad_(True)
    exp_grad_factors = torch.tensor(expected_gradient_factors).float().requires_grad_(True)

    assert isinstance(quantization, quantization_from_name(quantization.name))

    y = quantization(x)
    assert torch.allclose(y, x_exp, atol=0.001)

    y.backward(x)
    computed_gradient = x.grad.clone()
    expected_gradient = x * exp_grad_factors

    assert torch.allclose(computed_gradient, expected_gradient, atol=0.001)
