Module bitorch.quantizations.swish_sign
=======================================
Sign Function Implementation

Classes
-------

`SwishSign(beta: Optional[float] = None)`
:   Module for applying the SwishSign function
    
    Initializes gradient cancelation threshold.
    
    Args:
        gradient_cancelation_threshold (float, optional): threshold after which gradient is 0. Defaults to 1.0.

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
    :   Forwards the tensor through the swishsign function.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: sign of tensor x

`SwishSignFunction(*args, **kwargs)`
:   SwishSign Function for input binarization.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: torch.autograd.function.BackwardCFunction, output_grad: torch.Tensor) ‑> Tuple[torch.Tensor, None]`
    :   Apply straight through estimator.
        
        This passes the output gradient as input gradient after clamping the gradient values to the range [-1, 1]
        
        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient
        
        Returns:
            torch.Tensor: the input gradient (= the clamped output gradient)

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor, beta: float = 1.0) ‑> torch.Tensor`
    :   Binarize input tensor using the _sign function.
        
        Args:
            input_tensor (tensor): the input values to the Sign function
        
        Returns:
            tensor: binarized input tensor