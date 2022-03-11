Module bitorch.quantizations.sign
=================================
Sign Function Implementation

Classes
-------

`Sign()`
:   Module for applying the sign function with straight through estimator in backward pass
    
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
    :   Forwards the tensor through the sign function.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: sign of tensor x

`SignFunction(*args, **kwargs)`
:   Straight Through estimator for backward pass

    ### Ancestors (in MRO)

    * bitorch.quantizations.base.STE
    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   Binarize the input tensor using the sign function
        
        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: the sign tensor