from bitorch.layers import QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct
from torch.nn.functional import conv1d, conv2d, conv3d
from bitorch.quantizations import Sign
import pytest
import torch
import numpy as np

TEST_INPUT_DATA = [
    (QConv1d_NoAct, conv1d, (1, 2, 5), [2, 2], {"kernel_size": 3, "weight_quantization": "sign", "padding": 1}),
    (QConv2d_NoAct, conv2d, (1, 2, 5, 5), [2, 2], {"kernel_size": 3, "weight_quantization": "sign", "padding": 1}),
    (QConv3d_NoAct, conv3d, (1, 2, 4, 4, 4), [2, 2], {"kernel_size": 3, "weight_quantization": "sign", "padding": 1}),
    (QConv1d_NoAct, conv1d, (1, 2, 5), [2, 2], {"kernel_size": 3, "weight_quantization": Sign(), "padding": 1}),
    (QConv2d_NoAct, conv2d, (1, 2, 5, 5), [2, 2], {"kernel_size": 3, "weight_quantization": Sign(), "padding": 1}),
    (QConv3d_NoAct, conv3d, (1, 2, 4, 4, 4), [2, 2], {"kernel_size": 3, "weight_quantization": Sign(), "padding": 1}),
] * 10


@pytest.mark.parametrize("conv_layer, conv_fn, input_shape, args, kwargs", TEST_INPUT_DATA)
def test_qconv(conv_layer, conv_fn, input_shape, args, kwargs):
    input_values = np.sign(np.random.uniform(-1, 1, input_shape))
    input_values[input_values == 0] = 1
    input_tensor = torch.tensor(input_values).float().requires_grad_(True)
    layer = conv_layer(*args, **kwargs)

    result1 = layer(input_tensor)
    result1.backward(input_tensor)

    grad1 = input_tensor.grad.clone()
    input_tensor.grad.zero_()

    binary_weights = layer._weight_quantize(layer.weight.clone())

    padding = kwargs["padding"]
    dimensionality = len(input_shape) - 2
    padding_list = [padding] * 2 * dimensionality
    padded_input = torch.nn.functional.pad(input_tensor, padding_list, mode="constant", value=-1)
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

    assert torch.equal(padded_input, layer._apply_padding(input_tensor))
    assert torch.equal(result1, direct_result)
    assert torch.equal(grad1, grad2)
