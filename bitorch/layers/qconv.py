"""Module containing the quantized convolution layer"""

from typing import Type, Union
from torch import Tensor

from bitorch.quantizations import Quantization
from bitorch.layers.qactivation import QActivation
from bitorch.layers.qconv_noact import QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct


class QConv1d(QConv1d_NoAct):  # type: ignore
    def __init__(self,  # type: ignore
                    *args,  # type: ignore
                    input_quantization: Union[str, Quantization] = None,
                    weight_quantization: Union[str, Quantization] = None,
                    **kwargs) -> None:  # type: ignore
        """initialization function for quantization of inputs and weights.

        Args:
            input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function for weights. Defaults to None.
        """
        super(QConv1d, self).__init__(*args, weight_quantization=weight_quantization, **kwargs)
        self.activation = QActivation(input_quantization)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """forward the input tensor through the activation and quantized convolution layer.

        Args:
            input_tensor (Tensor): input tensor

        Returns:
            Tensor: the activated and convoluted output tensor.
        """
        return super(QConv1d, self).forward(self.activation(input_tensor))

class QConv2d(QConv2d_NoAct):  # type: ignore
    def __init__(self,  # type: ignore
                    *args,  # type: ignore
                    input_quantization: Union[str, Quantization] = None,
                    weight_quantization: Union[str, Quantization] = None,
                    **kwargs) -> None:  # type: ignore
        """initialization function for quantization of inputs and weights.

        Args:
            input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function for weights. Defaults to None.
        """
        super(QConv2d, self).__init__(*args, weight_quantization=weight_quantization, **kwargs)
        self.activation = QActivation(input_quantization)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """forward the input tensor through the activation and quantized convolution layer.

        Args:
            input_tensor (Tensor): input tensor

        Returns:
            Tensor: the activated and convoluted output tensor.
        """
        return super(QConv2d, self).forward(self.activation(input_tensor))


class QConv3d(QConv3d_NoAct):  # type: ignore
    def __init__(self,  # type: ignore
                    *args,  # type: ignore
                    input_quantization: Union[str, Quantization] = None,
                    weight_quantization: Union[str, Quantization] = None,
                    **kwargs) -> None:  # type: ignore
        """initialization function for quantization of inputs and weights.

        Args:
            input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function for weights. Defaults to None.
        """
        super(QConv3d, self).__init__(*args, weight_quantization=weight_quantization, **kwargs)
        self.activation = QActivation(input_quantization)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """forward the input tensor through the activation and quantized convolution layer.

        Args:
            input_tensor (Tensor): input tensor

        Returns:
            Tensor: the activated and convoluted output tensor.
        """
        return super(QConv3d, self).forward(self.activation(input_tensor))
