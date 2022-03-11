Module bitorch.quantizations
============================

Sub-modules
-----------
* bitorch.quantizations.approx_sign
* bitorch.quantizations.base
* bitorch.quantizations.config
* bitorch.quantizations.dorefa
* bitorch.quantizations.identity
* bitorch.quantizations.sign
* bitorch.quantizations.ste_heaviside
* bitorch.quantizations.swish_sign

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

`Identity()`
:   Module that provides the identity function, which can be useful for certain training strategies
    
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
    :   forwards the input tensor x without quantization.
        
        Args:
            x (torch.Tensor): tensor to be forwarded.
        
        Returns:
            torch.Tensor: tensor x

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