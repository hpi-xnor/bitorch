Module bitorch.layers
=====================

Sub-modules
-----------
* bitorch.layers.config
* bitorch.layers.debug_layers
* bitorch.layers.qactivation
* bitorch.layers.qconv
* bitorch.layers.qconv_noact
* bitorch.layers.qlinear

Classes
-------

`InputGraphicalDebug(figure: object = None, images: list = None, debug_interval: int = 100, num_outputs: int = 10)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    Debugs the given layer by drawing weights/inputs in given matplotlib plot images.
    
    Args:
        figure (object): figure to draw in
        images (list): list of images to update with given data
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
    
    Raises:
        ValueError: raised if number of images does not match desired number of outputs.

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._GraphicalDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the given tensor without modification, debug output if activated
        
        Args:
            x (torch.Tensor): tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`InputPrintDebug(debug_interval: int = 100, num_outputs: int = 10, name: str = 'Debug')`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    inits values.
    
    Args:
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
        name (str, optional): name of debug layer, only relevant for print debugging

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   forwards the given tensor without modification, debug output if activated
        
        Args:
            x (torch.Tensor): tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`QActivation(activation: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = 0.0)`
:   Activation layer for quantization
    
    initialization function for fetching suitable activation function.
    
    Args:
        activation (Union[str, Quantization], optional): quantization module or name of quantization function.
            Defaults to None.
        gradient_cancellation_threshold (Optional[float], optional): threshold for input gradient
            cancellation. Disabled if threshold is 0.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   Forwards input tensor through activation function.
        
        Args:
            input_tensor (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: quantized input tensor.

`QLinear(*args: int, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, **kwargs: bool)`
:   Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    
    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`
    
    Examples::
    
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    
    Applys the given quantization functions on weights and inputs before applying the linear operation.
    
    Args:
        *args (Argument list): positional arguments for linear layer
        input_quantization (Union[str, Quantization], optional): quantization module used for input
            quantization. Defaults to None.
        gradient_cancellation_threshold (Union[float, None], optional): threshold for input gradient cancellation.
            disabled if threshold is None. Defaults to None.
        weight_quantization (Union[str, Quantization], optional): quantization module or name of quantization
            function. Defaults to None.
        **kwargs (keyword Argument list): keyword arguments for linear layer

    ### Ancestors (in MRO)

    * torch.nn.modules.linear.Linear
    * torch.nn.modules.module.Module

    ### Class variables

    `in_features: int`
    :

    `out_features: int`
    :

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forwards x through the binary linear layer.
        
        Args:
            x (torch.Tensor): tensor to forward
        
        Returns:
            torch.Tensors: forwarded tensor

`ShapePrintDebug(debug_interval: int = 100, num_outputs: int = 10, name: str = 'Debug')`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    inits values.
    
    Args:
        debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
        num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
        name (str, optional): name of debug layer, only relevant for print debugging

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   prints the shape of x, leaves x untouched
        
        Args:
            x (torch.Tensor): the tensor to be debugged
        
        Returns:
            torch.Tensor: input tensor x

`WeightGraphicalDebug(module: torch.nn.modules.module.Module, *args, **kwargs)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    stores given module
    
    Args:
        module (torch.nn.Module): module the weights of which shall be debugged

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._GraphicalDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forwards the input tensor through the debug model and outputs debug information about the given modules weights.
        
        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.
        
        Returns:
            torch.Tensor: the input tensor

`WeightPrintDebug(module: torch.nn.modules.module.Module, *args, **kwargs)`
:   Base class for all neural network modules.
    
    Your models should also subclass this class.
    
    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::
    
        import torch.nn as nn
        import torch.nn.functional as F
    
        class Model(nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)
    
            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))
    
    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    
    :ivar training: Boolean represents whether this module is in training or
                    evaluation mode.
    :vartype training: bool
    
    stores given module
    
    Args:
        module (torch.nn.Module): module the weights of which shall be debugged

    ### Ancestors (in MRO)

    * bitorch.layers.debug_layers._PrintDebug
    * bitorch.layers.debug_layers._Debug
    * torch.nn.modules.module.Module

    ### Class variables

    `dump_patches: bool`
    :

    `training: bool`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forwards the input tensor through the debug model and outputs debug information about the given modules weights.
        
        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.
        
        Returns:
            torch.Tensor: the input tensor

`QConv1d(*args, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, **kwargs)`
:   Applies a 1D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:
    
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)
    
    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    
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
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    
    
    
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where
    
          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
    
    Examples::
    
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
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

    * bitorch.layers.qconv_noact.make_q_convolution_noact.<locals>.QConv_NoAct
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

`QConv2d(*args, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, **kwargs)`
:   Applies a 2D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:
    
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
    
    
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    
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
    
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    
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
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    
    
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
    
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
    
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    
    Examples:
    
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
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

    * bitorch.layers.qconv_noact.make_q_convolution_noact.<locals>.QConv_NoAct
    * torch.nn.modules.conv.Conv2d
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

`QConv3d(*args, input_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, gradient_cancellation_threshold: Optional[float] = None, **kwargs)`
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

    * bitorch.layers.qconv_noact.make_q_convolution_noact.<locals>.QConv_NoAct
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

`QConv1d_NoAct(*args, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, pad_value: float = None, bias: bool = False, **kwargs)`
:   Applies a 1D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:
    
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)
    
    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    
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
        padding (int, tuple or str, optional): Padding added to both sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel
            elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    
    
    
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`(N, C_{out}, L_{out})` where
    
          .. math::
              L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels},
            \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \text{kernel\_size}}`
    
    Examples::
    
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = torch.randn(20, 16, 50)
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

    * torch.nn.modules.conv.Conv1d
    * torch.nn.modules.conv._ConvNd
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.layers.qconv.make_q_convolution.<locals>.QConv

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

`QConv2d_NoAct(*args, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, pad_value: float = None, bias: bool = False, **kwargs)`
:   Applies a 2D convolution over an input signal composed of several input
    planes.
    
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:
    
    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)
    
    
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    
    
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    
    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    
    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {'valid', 'same'} or a tuple of ints giving the
      amount of implicit padding applied on both sides.
    
    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    
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
    
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    
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
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    
    
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
    
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
    
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    
    Examples:
    
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
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

    * torch.nn.modules.conv.Conv2d
    * torch.nn.modules.conv._ConvNd
    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.layers.qconv.make_q_convolution.<locals>.QConv

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

`QConv3d_NoAct(*args, weight_quantization: Union[str, bitorch.quantizations.base.Quantization] = None, pad_value: float = None, bias: bool = False, **kwargs)`
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

    * bitorch.layers.qconv.make_q_convolution.<locals>.QConv

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