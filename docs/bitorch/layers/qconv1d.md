Module bitorch.layers.qconv1d
=============================
Module containing the quantized convolution layer

Functions
---------

    
`conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) ‑> torch.Tensor`
:   Applies a 1D convolution over an input signal composed of several input
    planes.
    
    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    See :class:`~torch.nn.Conv1d` for details and output shape.
    
    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.
    
    
    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
        weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)`
        bias: optional bias of shape :math:`(\text{out\_channels})`. Default: ``None``
        stride: the stride of the convolving kernel. Can be a single number or
          a one-element tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
          single number or a one-element tuple `(padW,)`. Default: 0
          ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
          the input so the output has the shape as the input. However, this mode
          doesn't support any stride values other than 1.
    
          .. warning::
              For ``padding='same'``, if the ``weight`` is even-length and
              ``dilation`` is odd in any dimension, a full :func:`pad` operation
              may be needed internally. Lowering performance.
        dilation: the spacing between kernel elements. Can be a single number or
          a one-element tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
          the number of groups. Default: 1
    
    Examples::
    
        >>> inputs = torch.randn(33, 16, 30)
        >>> filters = torch.randn(20, 16, 5)
        >>> F.conv1d(inputs, filters)

Classes
-------

`QConv1d(*args: Any, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, **kwargs: Any)`
:   Quantized 1d Convolutional Layer. Has the same api as Conv1d but lets you specify a weight quantization, that is applied before the convolutional operation.
    
    initialization function for quantization of inputs and weights.
    
    Args:
        input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
        gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient
            cancellation. Disabled if threshold is None. Defaults to None.
        weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function for weights. Defaults to None.

    ### Ancestors (in MRO)

    * bitorch.layers.qconv1d.QConv1d_NoAct
    * torch.nn.modules.conv.Conv1d
    * torch.nn.modules.conv._ConvNd
    * torch.nn.modules.module.Module

    ### Class variables

    `bias: Optional[torch.Tensor]`
    :

    `dilation: Tuple[int, ...]`
    :

    `groups: int`
    :

    `kernel_size: Tuple[int, ...]`
    :

    `out_channels: int`
    :

    `output_padding: Tuple[int, ...]`
    :

    `padding: Union[str, Tuple[int, ...]]`
    :

    `padding_mode: str`
    :

    `stride: Tuple[int, ...]`
    :

    `transposed: bool`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   forward the input tensor through the activation and quantized convolution layer.
        
        Args:
            input_tensor (Tensor): input tensor
        
        Returns:
            Tensor: the activated and convoluted output tensor.

`QConv1d_NoAct(*args: Any, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, pad_value: float = None, bias: bool = False, **kwargs: Any)`
:   Quantized 1d Convolutional Layer. Has the same api as Conv1d but lets you specify a weight quantization, that is applied before the convolutional operation.
    
    initialization function for padding and quantization.
    
    Args:
        weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function. Defaults to None.
        padding_value (float, optional): value used for padding the input sequence. Defaults to None.

    ### Ancestors (in MRO)

    * torch.nn.modules.conv.Conv1d
    * torch.nn.modules.conv._ConvNd
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.layers.qconv1d.QConv1d

    ### Class variables

    `bias: Optional[torch.Tensor]`
    :

    `dilation: Tuple[int, ...]`
    :

    `groups: int`
    :

    `kernel_size: Tuple[int, ...]`
    :

    `out_channels: int`
    :

    `output_padding: Tuple[int, ...]`
    :

    `padding: Union[str, Tuple[int, ...]]`
    :

    `padding_mode: str`
    :

    `stride: Tuple[int, ...]`
    :

    `transposed: bool`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, input: torch.Tensor) ‑> torch.Tensor`
    :   forward the input tensor through the quantized convolution layer.
        
        Args:
            input (Tensor): input tensor
        
        Returns:
            Tensor: the convoluted output tensor, computed with padded input and quantized weights.

    `reset_parameters(self) ‑> None`
    :   overwritten from _ConvNd to initialize weights