Module bitorch.quantizations.ste_heaviside
==========================================
Sign Function Implementation

Classes
-------

`SteHeaviside()`
:   Module for applying the SteHeaviside quantization, using an ste in backward pass
    
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

`SteHeavisideFunction(*args, **kwargs)`
:   Straight Through estimator for backward pass

    ### Ancestors (in MRO)

    * bitorch.quantizations.base.STE
    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Static methods

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   quantizes input tensor and forwards it.
        
        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: the quantized input tensor