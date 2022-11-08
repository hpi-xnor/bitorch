"""Module containing the quantized 3d convolution layer"""

from typing import Optional, Any, Type, Union

from torch import Tensor
from torch.nn import Conv3d, init
from torch.nn.functional import pad, conv3d

from bitorch import RuntimeMode
from bitorch.quantizations import Quantization
from .config import config
from .extensions import DefaultImplementationMixin
from .qactivation import QActivation
from .qconv_mixin import QConvArgsProviderMixin
from .register import QConv3dImplementation


class QConv3d_NoAct(Conv3d):  # type: ignore # noqa: N801
    def __init__(
        self,  # type: ignore
        *args: Any,
        weight_quantization: Optional[Union[str, Quantization]] = None,
        pad_value: Optional[float] = None,
        bias: bool = False,
        **kwargs: Any,
    ) -> None:
        """initialization function for padding and quantization.

        Args:
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function. Defaults to None.
            padding_value (float, optional): value used for padding the input sequence. Defaults to None.
        """
        assert bias is False, "A QConv layer can not use a bias due to acceleration techniques during deployment."
        kwargs["bias"] = False
        super(QConv3d_NoAct, self).__init__(*args, **kwargs)
        self._weight_quantize = config.get_quantization_function(weight_quantization or config.weight_quantization)
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
        return conv3d(  # type: ignore
            input=self._apply_padding(input),
            weight=self._weight_quantize(self.weight),
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class QConv3dBase(QConvArgsProviderMixin, QConv3d_NoAct):  # type: ignore
    def __init__(
        self,  # type: ignore
        *args: Any,
        input_quantization: Optional[Union[str, Quantization]] = None,
        weight_quantization: Optional[Union[str, Quantization]] = None,
        gradient_cancellation_threshold: Union[float, None] = None,
        **kwargs: Any,
    ) -> None:
        """initialization function for quantization of inputs and weights.

        Args:
            input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
            gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient
                cancellation. Disabled if threshold is None. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function for weights. Defaults to None.
        """
        super().__init__(*args, weight_quantization=weight_quantization, **kwargs)
        self.activation = QActivation(input_quantization, gradient_cancellation_threshold)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """forward the input tensor through the activation and quantized convolution layer.

        Args:
            input_tensor (Tensor): input tensor

        Returns:
            Tensor: the activated and convoluted output tensor.
        """
        return super().forward(self.activation(input_tensor))


class _QConv3dComposed(DefaultImplementationMixin, QConv3dBase):
    """
    This class defines the default implementation of a QConv3d layer (which is actually implemented by QConv3dBase).

    To implement a custom QConv3d implementation use QConv3dBase as a super class instead.
    """

    pass


QConv3d: Type[_QConv3dComposed] = QConv3dImplementation(RuntimeMode.DEFAULT)(_QConv3dComposed)  # type: ignore
"""
This class provides the current implementation of a QConv3d layer (which is actually implemented by :class:`QConv3dBase`).

To implement a custom QConv3d implementation use :class:`QConv3dBase` as a super class instead.
"""
