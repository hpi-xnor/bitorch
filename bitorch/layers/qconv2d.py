"""Module containing the quantized 2d convolution layer"""

from typing import Union, Any, Dict

from torch import Tensor
from torch.nn import Conv2d, init
from torch.nn.functional import pad, conv2d

from bitorch import RuntimeMode
from bitorch.quantizations import Quantization
from .config import config
from .extensions import DefaultImplementationMixin, LayerRecipe
from .qactivation import QActivation
from .register import QConv2dImplementation


class QConv2d_NoAct(Conv2d):  # type: ignore # noqa: N801
    def __init__(self,  # type: ignore
                 *args: Any,
                 weight_quantization: Union[str, Quantization] = None,
                 pad_value: float = None,
                 bias: bool = False,
                 **kwargs: Any) -> None:
        """initialization function for padding and quantization.

        Args:
            weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
                function. Defaults to None.
            padding_value (float, optional): value used for padding the input sequence. Defaults to None.
        """
        assert bias is False, "A QConv layer can not use a bias due to acceleration techniques during deployment."
        kwargs["bias"] = False
        super(QConv2d_NoAct, self).__init__(*args, **kwargs)
        self._weight_quantize = config.get_quantization_function(
            weight_quantization or config.weight_quantization)
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
        return conv2d(  # type: ignore
            input=self._apply_padding(input),
            weight=self._weight_quantize(self.weight),
            bias=None,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups)


class QConv2dBase(QConv2d_NoAct):  # type: ignore
    def __init__(self,  # type: ignore
                 *args: Any,
                 input_quantization: Union[str, Quantization] = None,
                 weight_quantization: Union[str, Quantization] = None,
                 gradient_cancellation_threshold: Union[float, None] = None,
                 **kwargs: Any) -> None:
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

    @staticmethod
    def get_args_as_kwargs(recipe: LayerRecipe) -> Dict[str, Any]:
        """
        Gather all arguments that were used to create a QLinear layer with argument names.
        Can be used to recreate a layer with identical arguments.

        Returns:
            A dictionary with all arguments (key is the argument name as a string even for positional arguments)
        """
        _ = ...
        return {
            # "in_features": recipe.get_positional_arg(0),
            # "out_features": recipe.get_positional_arg(1),
            # "input_quantization": recipe.layer.input_quantization,
            # "gradient_cancellation_threshold": recipe.layer.gradient_cancellation_threshold,
            # "weight_quantization": recipe.layer.weight_quantization,
            # "bias": recipe.get_arg(5, "bias", True),
            "in_channels": _,
            "out_channels": _,
            "kernel_size": _,
            "stride": _,
            "padding": _,
            "dilation": _,
            "transposed": _,
            "output_padding": _,
            "groups": _,
            "bias": _,
            "padding_mode": _,
            "device": recipe.get_arg(6, "device", None),
            "dtype": recipe.get_arg(7, "dtype", None),
        }

    def forward(self, input_tensor: Tensor) -> Tensor:
        """forward the input tensor through the activation and quantized convolution layer.

        Args:
            input_tensor (Tensor): input tensor

        Returns:
            Tensor: the activated and convoluted output tensor.
        """
        return super().forward(self.activation(input_tensor))


@QConv2dImplementation(RuntimeMode.DEFAULT)
class QConv2d(DefaultImplementationMixin, QConv2dBase):
    """
    This class defines the default implementation of a QConv2d layer (which is actually implemented by QConv2dBase).

    To implement a custom QConv2d implementation use QConv2dBase as a super class instead.
    """
    pass
