from bitorch.layers import QConv1d, QConv2d, QConv3d
from torch.nn.functional import conv1d, conv2d, conv3d
from bitorch.quantizations import Sign
import pytest
import torch
import numpy as np

TEST_INPUT_DATA = [
    (
        QConv1d,
        conv1d,
        (1, 2, 5),
        [2, 2],
        {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 1},
    ),
    (
        QConv2d,
        conv2d,
        (1, 2, 5, 5),
        [2, 2],
        {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 1},
    ),
    (
        QConv3d,
        conv3d,
        (1, 2, 4, 4, 4),
        [2, 2],
        {"kernel_size": 3, "weight_quantization": "sign", "input_quantization": "sign", "padding": 1},
    ),
    (
        QConv1d,
        conv1d,
        (1, 2, 5),
        [2, 2],
        {
            "kernel_size": 3,
            "weight_quantization": Sign(),
            "input_quantization": "sign",
            "gradient_cancellation_threshold": 0.5,
            "padding": 1,
        },
    ),
    (
        QConv2d,
        conv2d,
        (1, 2, 5, 5),
        [2, 2],
        {
            "kernel_size": 3,
            "weight_quantization": Sign(),
            "input_quantization": "sign",
            "gradient_cancellation_threshold": 1.0,
            "padding": 1,
        },
    ),
    (
        QConv3d,
        conv3d,
        (1, 2, 4, 4, 4),
        [2, 2],
        {
            "kernel_size": 3,
            "weight_quantization": Sign(),
            "input_quantization": "sign",
            "gradient_cancellation_threshold": 2.0,
            "padding": 1,
        },
    ),
]


@pytest.mark.parametrize("execution_number", range(10))
@pytest.mark.parametrize("conv_layer, conv_fn, input_shape, args, kwargs", TEST_INPUT_DATA)
def test_qconv(conv_layer, conv_fn, input_shape, args, kwargs, execution_number):
    input_values = np.random.uniform(-1, 1, input_shape)
    layer = conv_layer(*args, **kwargs)
    input_tensor = torch.tensor(input_values).float().requires_grad_(True)

    result1 = layer(input_tensor)
    result1.backward(input_tensor)

    grad1 = input_tensor.grad.clone()
    input_tensor.grad.zero_()

    binary_weights = layer._weight_quantize(layer.weight.clone())

    expected_tensor = layer.activation(input_tensor).requires_grad_(True)
    padding = kwargs["padding"]
    dimensionality = len(input_shape) - 2
    padding_list = [padding] * 2 * dimensionality
    padded_input = torch.nn.functional.pad(expected_tensor, padding_list, mode="constant", value=-1)
    direct_result = conv_fn(
        input=padded_input,
        weight=binary_weights,
        bias=layer.bias,
        stride=layer.stride,
        padding=0,
        dilation=layer.dilation,
        groups=layer.groups,
    )
    direct_result.backward(input_tensor)
    grad2 = input_tensor.grad.clone()

    assert torch.equal(padded_input, layer._apply_padding(expected_tensor))
    assert torch.equal(result1, direct_result)
    assert torch.equal(grad1, grad2)
