"""Module containing the quantized convolution layer"""

from typing import Callable, Type
from torch import Tensor
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from . import layerconfig
from torch.nn.functional import pad, conv1d, conv2d, conv3d


def make_q_convolution(BaseClass: Type, forward_fn: Callable) -> Type:
    """Creates a quantized version of the given convolution base class.

    Args:
        BaseClass (QConv-Subclass): The base class to create a quantized version from.
        forward_fn (function): the convolution function used for this class.

    Returns:
        Class: the quantized version of the base class
    """
    class QConv(BaseClass):  # type: ignore
        def __init__(self, *args, quantization: str = None, pad_value: float = None, **kwargs) -> None:  # type: ignore
            """initialization function for padding and quantization.

            Args:
                quantization (str, optional): name of quantization function. Defaults to None.
                padding_value (float, optional): value used for padding the input sequence. Defaults to None.
            """
            super(QConv, self).__init__(*args, **kwargs)
            self.quantize = layerconfig.config.get_quantization_function(quantization)
            self.pad_value = pad_value or layerconfig.config.get_padding_value()

        def _apply_padding(self, x: Tensor) -> Tensor:
            """pads the input tensor with the given padding value

            Args:
                x (Tensor): input tensor

            Returns:
                Tensor: the padded tensor
            """
            return pad(x, self._reversed_padding_repeated_twice, mode="constant", value=self.pad_value)

        def forward(self, input: Tensor) -> Tensor:
            """forward the input tensor through the quantized convolution layer.

            Args:
                input (Tensor): input tensor
                weight (Tensor): weight tensor
                bias (Tensor, optional): bias tensor. Defaults to None.

            Returns:
                Tensor: the convoluted output tensor, computed with padded input and quantized weights.
            """
            return forward_fn(  # type: ignore
                input=self._apply_padding(input),
                weight=self.quantize(self.weight),
                bias=self.bias,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups)

    return QConv


QConv1d = make_q_convolution(Conv1d, conv1d)
QConv2d = make_q_convolution(Conv2d, conv2d)
QConv3d = make_q_convolution(Conv3d, conv3d)
