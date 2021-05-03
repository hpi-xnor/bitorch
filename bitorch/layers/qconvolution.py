"""Module containing the quantized convolution layer"""

from torch import Tensor
from torch.nn.modules.conv import Conv1d, Conv2d, Conv3d
from . import layerconfig
from torch.nn.functional import pad, conv1d, conv2d, conv3d


class QConvolution1d(Conv1d):
    """Quantized 1d concolution class"""

    def __init__(self, *args, quantization: str = None, padding_value: float = None, **kwargs) -> None:  # type: ignore
        """initialization function for padding and quantization.

        Args:
            quantization (str, optional): name of quantization function. Defaults to None.
            padding_value (float, optional): value used for padding the input sequence. Defaults to None.
        """
        super(QConvolution1d, self).__init__(*args, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)
        self.padding_value = padding_value or layerconfig.config.get_padding_value()

    def _apply_padding(self, x: Tensor) -> Tensor:
        """pads the input tensor with the given padding value

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: the padded tensor
        """
        return pad(x, self._reversed_padding_repeated_twice, mode="constant", value=self.padding_value)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
        """forward the input tensor through the quantized 1d convolution layer.

        Args:
            input (Tensor): input tensor
            weight (Tensor): weight tensor
            bias (Tensor, optional): bias tensor. Defaults to None.

        Returns:
            Tensor: the convoluted output tensor, computed with padded input and quantized weights.
        """
        return conv1d(
            input=self._apply_padding(input),
            weight=self.quantize(weight),
            bias=bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups)


class QConvolution2d(Conv2d):
    """Quantized 2d concolution class"""

    def __init__(self, *args, quantization: str = None, padding_value: float = None, **kwargs) -> None:  # type: ignore
        """initialization function for padding and quantization.

        Args:
            quantization (str, optional): name of quantization function. Defaults to None.
            padding_value (float, optional): value used for padding the input sequence. Defaults to None.
        """
        super(QConvolution2d, self).__init__(*args, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)
        self.padding_value = padding_value or layerconfig.config.get_padding_value()

    def _apply_padding(self, x: Tensor) -> Tensor:
        """pads the input tensor with the given padding value

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: the padded tensor
        """
        return pad(x, self._reversed_padding_repeated_twice, mode="constant", value=self.padding_value)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
        """forward the input tensor through the quantized 2d convolution layer.

        Args:
            input (Tensor): input tensor
            weight (Tensor): weight tensor
            bias (Tensor, optional): bias tensor. Defaults to None.

        Returns:
            Tensor: the convoluted output tensor, computed with padded input and quantized weights.
        """
        return conv2d(
            input=self._apply_padding(input),
            weight=self.quantize(weight),
            bias=bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups)


class QConvolution3d(Conv3d):
    """Quantized 3d concolution class"""

    def __init__(self, *args, quantization: str = None, padding_value: float = None, **kwargs) -> None:  # type: ignore
        """initialization function for padding and quantization.

        Args:
            quantization (str, optional): name of quantization function. Defaults to None.
            padding_value (float, optional): value used for padding the input sequence. Defaults to None.
        """
        super(QConvolution3d, self).__init__(*args, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)
        self.padding_value = padding_value or layerconfig.config.get_padding_value()

    def _apply_padding(self, x: Tensor) -> Tensor:
        """pads the input tensor with the given padding value

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: the padded tensor
        """
        return pad(x, self._reversed_padding_repeated_twice, mode="constant", value=self.padding_value)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
        """forward the input tensor through the quantized 3d convolution layer.

        Args:
            input (Tensor): input tensor
            weight (Tensor): weight tensor
            bias (Tensor, optional): bias tensor. Defaults to None.

        Returns:
            Tensor: the convoluted output tensor, computed with padded input and quantized weights.
        """
        return conv3d(
            input=self._apply_padding(input),
            weight=self.quantize(weight),
            bias=bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups)
