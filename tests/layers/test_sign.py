from math import exp
from bitorch.layers.Sign import SignFunction
import torch
from torch.autograd import gradcheck
import pytest

sign = SignFunction()

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

TEST_OUTPUT_DATA = [
    ([1, 2, 3, 4, 5, 6], [1] * 6),
    ([-1, -2, -3, -4, -5, -6], [-1] * 6),
    ([0], [0]),
    ([-0.5, -0.1, 0.1, -0.5], [-0.5, -0.1, 0.1, -0.5]),
    ([1e-10, 1e-11, 1e-12, -1e-10, -1e-11, -1e-12, 0], [1e-10, 1e-11, 1e-12, -1e-10, -1e-11, -1e-12, 0]),
    ([1e10, 1e11, 1e12, -1e10, -1e11, -1e12, 0], [1, 1, 1, -1, -1, -1, 0]),
]


@pytest.mark.parametrize("input_values, expected_output", TEST_INPUT_DATA)
def test_forward_pass(input_values, expected_output):
    assert torch.equal(sign.forward(None, torch.Tensor(input_values)), torch.Tensor(expected_output))


@pytest.mark.parametrize("input_values, expected_output", TEST_OUTPUT_DATA)
def test_backward_pass(input_values, expected_output):
    assert torch.equal(sign.backward(None, torch.Tensor(input_values)), torch.Tensor(expected_output))
