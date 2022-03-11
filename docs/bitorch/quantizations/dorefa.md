Module bitorch.quantizations.dorefa
===================================
Dorefa Function Implementation

Classes
-------

`InputDoReFa(bits: Optional[int] = None)`
:   Module for applying the dorefa function on inputs.
    
    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    
    Initiates quantization bits.
    
    Args:
        bits (int, optional): number of bits to quantize into. Defaults to None.

    ### Ancestors (in MRO)

    * bitorch.quantizations.base.Quantization
    * torch.nn.modules.module.Module

    ### Class variables

    `bitwidth`
    :

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

    ### Methods

    `quantize(self, x: torch.Tensor) ‑> torch.Tensor`
    :   DoReFas the tensor to desired bit resolution.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: DoReFaed tensor x

`InputDoReFaFunction(*args, **kwargs)`
:   Base class to create custom `autograd.Function`
    
    To create a custom `autograd.Function`, subclass this class and implement
    the :meth:`forward` and :meth`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.
    
    To ensure correctness and best performance, make sure you are calling the
    correct methods on ``ctx`` and validating your backward function using
    :func:`torch.autograd.gradcheck`.
    
    See :ref:`extending-autograd` for more details on how to use this class.
    
    Examples::
    
        >>> class Exp(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> # Use it by calling the apply method:
        >>> output = Exp.apply(input)

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: Any, output_gradient: torch.Tensor) ‑> Tuple[torch.Tensor, None]`
    :   just passes the unchanged output gradient as input gradient (i.e. applies straight through estimator)
        
        Args:
            ctx (Any): autograd context
            output_gradient (torch.Tensor): output gradient
        
        Returns:
            torch.Tensor: the unchanged output gradient

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor, bits: int) ‑> torch.Tensor`
    :   quantizes input tensor and forwards it.
        
        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
            bits (int): number of bits to round the input tensor to
        
        Returns:
            torch.Tensor: the quantized input tensor

`WeightDoReFa(bits: Optional[int] = None)`
:   Module for applying the dorefa function on weights.
    
    Reference: "DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients"
    Zouh et al. 2016, https://arxiv.org/abs/1606.06160
    
    Initiates quantization bits.
    
    Args:
        bits (int, optional): number of bits to quantize into. Defaults to None.

    ### Ancestors (in MRO)

    * bitorch.quantizations.base.Quantization
    * torch.nn.modules.module.Module

    ### Class variables

    `bitwidth`
    :

    `dump_patches: bool`
    :

    `name`
    :

    `training: bool`
    :

    ### Methods

    `quantize(self, x: torch.Tensor) ‑> torch.Tensor`
    :   DoReFas the tensor to desired bit resolution using weight dorefa.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: DoReFaed tensor x