Module bitorch.layers.qconv3d
=============================
Module containing the quantized convolution layer

Functions
---------

    
`conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) ‑> torch.Tensor`
:   Applies a 3D convolution over an input image composed of several input
    planes.
    
    This operator supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    See :class:`~torch.nn.Conv3d` for details and output shape.
    
    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.
    
    
    Args:
        input: input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iT , iH , iW)`
        weight: filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kT , kH , kW)`
        bias: optional bias tensor of shape :math:`(\text{out\_channels})`. Default: None
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sT, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},
          single number or a tuple `(padT, padH, padW)`. Default: 0
          ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
          the input so the output has the shape as the input. However, this mode
          doesn't support any stride values other than 1.
    
          .. warning::
              For ``padding='same'``, if the ``weight`` is even-length and
              ``dilation`` is odd in any dimension, a full :func:`pad` operation
              may be needed internally. Lowering performance.
    
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dT, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by
          the number of groups. Default: 1
    
    Examples::
    
        >>> filters = torch.randn(33, 16, 3, 3, 3)
        >>> inputs = torch.randn(20, 16, 50, 10, 20)
        >>> F.conv3d(inputs, filters)

Classes
-------

`QConv3d(*args: Any, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, **kwargs: Any)`
:   Applies a 3D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:
    
    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)
    
    where :math:`\star` is the valid 3D `cross-correlation`_ operator
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
    
    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,
    
        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).
    
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
    
        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension
    
    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".
    
        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    
    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.
    
    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    
    
    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
    
          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
    
          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
    
    Examples::
    
        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)
    
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    initialization function for quantization of inputs and weights.
    
    Args:
        input_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function to apply on inputs before forwarding through the qconvolution layer. Defaults to None.
        gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient
            cancellation. Disabled if threshold is None. Defaults to None.
        weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function for weights. Defaults to None.

    ### Ancestors (in MRO)

    * bitorch.layers.qconv3d.QConv3d_NoAct
    * torch.nn.modules.conv.Conv3d
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

`QConv3d_NoAct(*args: Any, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, pad_value: float = None, bias: bool = False, **kwargs: Any)`
:   Applies a 3D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`
    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:
    
    .. math::
        out(N_i, C_{out_j}) = bias(C_{out_j}) +
                                \sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \star input(N_i, k)
    
    where :math:`\star` is the valid 3D `cross-correlation`_ operator
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.
    
    * :attr:`groups` controls the connections between inputs and outputs.
      :attr:`in_channels` and :attr:`out_channels` must both be divisible by
      :attr:`groups`. For example,
    
        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out\_channels}}{\text{in\_channels}}`).
    
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
    
        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension
    
    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".
    
        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.
    
    Note:
        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.
    
    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all six sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    
    
    Shape:
        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` where
    
          .. math::
              D_{out} = \left\lfloor\frac{D_{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
    
          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    
          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel\_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]}, \text{kernel\_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{2}\text{kernel\_size}[i]}`
    
    Examples::
    
        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> output = m(input)
    
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    
    initialization function for padding and quantization.
    
    Args:
        weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function. Defaults to None.
        padding_value (float, optional): value used for padding the input sequence. Defaults to None.

    ### Ancestors (in MRO)

    * torch.nn.modules.conv.Conv3d
    * torch.nn.modules.conv._ConvNd
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.layers.qconv3d.QConv3d

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