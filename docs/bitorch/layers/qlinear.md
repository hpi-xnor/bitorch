Module bitorch.layers.qlinear
=============================
Module containting the quantized linear layer

Classes
-------

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