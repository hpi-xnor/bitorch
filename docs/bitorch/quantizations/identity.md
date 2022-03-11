Module bitorch.quantizations.identity
=====================================
Identity Implementation

Classes
-------

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