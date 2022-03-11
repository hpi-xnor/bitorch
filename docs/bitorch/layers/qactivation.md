Module bitorch.layers.qactivation
=================================

Classes
-------

`GradientCancellation(*args, **kwargs)`
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

    `backward(ctx: torch.autograd.function.BackwardCFunction, output_grad: torch.Tensor) ‑> Tuple[torch.Tensor, None]`
    :   Apply straight through estimator.
        
        This passes the output gradient towards the input if the inputs are in the range [-1, 1].
        
        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient
        
        Returns:
            torch.Tensor: the input gradient (= the masked output gradient)

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor, threshold: float) ‑> torch.Tensor`
    :   Binarize input tensor using the _sign function.
        
        Args:
            input_tensor (tensor): the input values to the Sign function
        
        Returns:
            tensor: binarized input tensor

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