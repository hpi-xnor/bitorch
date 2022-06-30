import pytest

from bitorch.layers import (
    QConv1d,
    QConv1dBase,
    QConv2d,
    QConv2dBase,
    QConv3d,
    QConv3dBase,
    QLinear,
    QLinearBase,
)
from bitorch.quantizations import Sign


Q_CONV_ARGS = [
    ("in_channels", 16),
    ("out_channels", 64),
    ("kernel_size", 3),
    ("stride", 1),
    ("padding", 1),
    ("dilation", 1),
    ("groups", 1),
    ("bias", False),
    ("padding_mode", "zeros"),
    ("device", None),
    ("dtype", None),
]
Q_LINEAR_ARGS = [
    ("in_features", 64),
    ("out_features", 32),
    ("input_quantization", Sign()),
    ("gradient_cancellation_threshold", 1.3),
    ("weight_quantization", Sign()),
    ("bias", False),
    ("device", None),
    ("dtype", None),
]


@pytest.mark.parametrize(
    "all_args, layer, base_layer, num_positional_args",
    [
        [Q_CONV_ARGS, QConv1d, QConv1dBase, 3],
        [Q_CONV_ARGS, QConv2d, QConv2dBase, 3],
        [Q_CONV_ARGS, QConv3d, QConv3dBase, 3],
        [Q_LINEAR_ARGS, QLinear, QLinearBase, 2],
    ],
)
def test_args_function(all_args, layer, base_layer, num_positional_args: int):
    expected_result = {}
    layer_args = []
    layer_kwargs = {}

    for j, (key, val) in enumerate(all_args):
        expected_result[key] = val
        if j < num_positional_args:
            layer_args.append(val)
        else:
            layer_kwargs[key] = val

    layer = layer(*layer_args, **layer_kwargs)
    result = base_layer.get_args_as_kwargs(layer.recipe)
    assert result.keys() == expected_result.keys()
    for k in expected_result.keys():
        assert expected_result[k] == result[k]
