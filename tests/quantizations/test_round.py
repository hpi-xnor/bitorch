from bitorch.quantizations import DoReFa
import torch
import pytest

round_ste = DoReFa(bits=2)

TEST_INPUT_DATA = [
    ([0.89, 0.90, 0.95, 1.0, 2.0, 4.0], [1] * 6),
    ([0.11, 0.10, 0.05, 0.0, -1.0, -2.0], [0] * 6),
    ([-10.0, 0.30, 0.70, 1000.9], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]),
]


@pytest.mark.parametrize("input_values, expected_output", TEST_INPUT_DATA)
def test_sign_function(input_values: list, expected_output: list) -> None:
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_exp = torch.tensor(expected_output).float().requires_grad_(True)

    y = round_ste(x)
    assert torch.equal(y, x_exp)

    y.backward(x)
    computed_gradient = x.grad.clone()

    assert torch.equal(computed_gradient, x)
