"""Module containing the quantized convolution layer"""

from typing import Callable, Type, Union
from torch import Tensor
from torch.nn import init
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from torch.nn.functional import pad, conv1d, conv2d, conv3d

from bitorch.quantizations import Quantization
from .config import config


def make_q_convolution_noact(base_class: Type, forward_fn: Callable) -> Type:
    """Creates a quantized version of the given convolution base class.

    Args:
        base_class (QConv-Subclass): The base class to create a quantized version from.
        forward_fn (function): the convolution function used for this class.

    Returns:
        Class: the quantized version of the base class
    """
    class QConv_NoAct(base_class):  # type: ignore # noqa: N801
        def __init__(self,  # type: ignore
                     *args,  # type: ignore
                     weight_quantization: Union[str, Quantization] = None,
                     pad_value: float = None,
                     bias: bool = False,
                     **kwargs) -> None:  # type: ignore
            """initialization function for padding and quantization.

            Args:
                weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                    function. Defaults to None.
                padding_value (float, optional): value used for padding the input sequence. Defaults to None.
            """
            assert bias is False, "A QConv layer can not use a bias due to acceleration techniques during deployment."
            kwargs["bias"] = False
            super(QConv_NoAct, self).__init__(*args, **kwargs)
            self._weight_quantize = config.get_quantization_function(
                weight_quantization or config.weight_quantization())
            self._pad_value = pad_value or config.padding_value

        def _apply_padding(self, x: Tensor) -> Tensor:
            """pads the input tensor with the given padding value

            Args:
                x (Tensor): input tensor

            Returns:
                Tensor: the padded tensor
            """
            return pad(x, self._reversed_padding_repeated_twice, mode="constant", value=self._pad_value)

        def reset_parameters(self) -> None:
            """overwritten from _ConvNd to initialize weights"""
            init.xavier_normal_(self.weight)

        def forward(self, input: Tensor) -> Tensor:
            """forward the input tensor through the quantized convolution layer.

            Args:
                input (Tensor): input tensor

            Returns:
                Tensor: the convoluted output tensor, computed with padded input and quantized weights.
            """
            return forward_fn(  # type: ignore
                input=self._apply_padding(input),
                weight=self._weight_quantize(self.weight),
                bias=None,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups)

    return QConv_NoAct


QConv1d_NoAct = make_q_convolution_noact(Conv1d, conv1d)
QConv2d_NoAct = make_q_convolution_noact(Conv2d, conv2d)
QConv3d_NoAct = make_q_convolution_noact(Conv3d, conv3d)
