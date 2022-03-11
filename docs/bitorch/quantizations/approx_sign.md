Module bitorch.quantizations.approx_sign
========================================
Sign Function Implementation

Classes
-------

`ApproxSign()`
:   Module for applying the sign function with approx sign in backward pass
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

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
    :   Forwards the tensor through the approx sign function.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: sign of tensor x

`ApproxSignFunction(*args, **kwargs)`
:   ApproxSign Function for input binarization.

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `backward(ctx: torch.autograd.function.BackwardCFunction, output_grad: torch.Tensor) ‑> torch.Tensor`
    :   Apply approx sign function. used e.g. for birealnet
        
        Args:
            ctx (gradient context): context
            output_grad (toch.Tensor): the tensor containing the output gradient
        
        Returns:
            torch.Tensor: the input gradient

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   Binarize input tensor using the _sign function.
        
        Args:
            input_tensor (tensor): the input values to the Sign function
        
        Returns:
            tensor: binarized input tensor