from bitorch.layers.Sign import Sign
import torch
import pytest

sign = Sign()

TEST_INPUT_DATA = [
    ([1, 2, 3, 4, 5, 6], [1] * 6),
    ([-1, -2, -3, -4, -5, -6], [-1] * 6),
    ([0], [1]),
    ([1e10, 1e11, 1e12, -1e10, -1e11, -1e12, 0], [1, 1, 1, -1, -1, -1, 1]),
    ([1e-10, 1e-11, 1e-12, -1e-10, -1e-11, -1e-12, 0], [1, 1, 1, -1, -1, -1, 1]),
    ([[2, -2, 0, -1e-15], [-30, 42, 3e10, 0], [0, 0, 0, 0]], [[1, -1, 1, -1], [-1, 1, 1, 1], [1, 1, 1, 1]]),
    ([[3], [-3], [0]], [[1], [-1], [1]]),
    ([[[3]], [[-3]], [[0]]], [[[1]], [[-1]], [[1]]])
]


@pytest.mark.parametrize("input_values, expected_output", TEST_INPUT_DATA)
@pytest.mark.parametrize("threshold", [0.1, 0.5, 1.0, 2.0, 4.0])
def test_sign_function(input_values: list, expected_output: list, threshold: float) -> None:
    x = torch.tensor(input_values).float().requires_grad_(True)
    x_exp = torch.tensor(expected_output).float().requires_grad_(True)

    y = sign(x, threshold)
    assert torch.equal(y, x_exp)

    # now test gradient cancellation
    y.backward(x)
    computed_gradient = x.grad.clone()

    preactivation_input = x.clone().detach().requires_grad_(False)
    preactivation_input[abs(preactivation_input) >= threshold] = 0
    # preactivation_input = torch.clamp(preactivation_input, min=-1., max=1.)
    assert torch.equal(computed_gradient, preactivation_input)
    assert False
