Module bitorch.quantizations.base
=================================
Quantization superclass implementation

Classes
-------

`Quantization()`
:   superclass for quantization modules
    
    Initializes internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Descendants

    * bitorch.quantizations.approx_sign.ApproxSign
    * bitorch.quantizations.dorefa.InputDoReFa
    * bitorch.quantizations.dorefa.WeightDoReFa
    * bitorch.quantizations.identity.Identity
    * bitorch.quantizations.sign.Sign
    * bitorch.quantizations.ste_heaviside.SteHeaviside
    * bitorch.quantizations.swish_sign.SwishSign

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

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Quantizes the tensor using this classes quantize-method. Subclasses shall add some semantic there.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: quantized tensor x

    `quantize(self, x: torch.Tensor) ‑> torch.Tensor`
    :   quantize the input tensor. It is recommended to use a torch.Function to also maniputlate backward behaiviour. See
        the implementations of sign or dorefa quantization functions for more examples.
        
        Args:
            x (torch.Tensor): the input to be quantized
        
        Raises:
            NotImplementedError: raised if quantize function of superclass is called.
        
        Returns:
            torch.Tensor: the quantized tensor

`STE(*args, **kwargs)`
:   Straight Through estimator for backward pass

    ### Ancestors (in MRO)

    * torch.autograd.function.Function
    * torch._C._FunctionBase
    * torch.autograd.function.FunctionCtx
    * torch.autograd.function._HookMixin

    ### Descendants

    * bitorch.quantizations.sign.SignFunction
    * bitorch.quantizations.ste_heaviside.SteHeavisideFunction

    ### Static methods

    `backward(ctx: Any, output_gradient: torch.Tensor) ‑> torch.Tensor`
    :   just passes the unchanged output gradient as input gradient.
        
        Args:
            ctx (Any): autograd context
            output_gradient (torch.Tensor): output gradient
        
        Returns:
            torch.Tensor: the unchanged output gradient

    `forward(ctx: torch.autograd.function.BackwardCFunction, input_tensor: torch.Tensor) ‑> torch.Tensor`
    :   just fowards the unchanged input_tensor.
        
        Args:
            ctx (Any): autograd context
            input_tensor (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: the unchanged input tensor