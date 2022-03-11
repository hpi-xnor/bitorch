Module bitorch.models.lenet
===========================

Classes
-------

`LeNet(dataset: bitorch.datasets.base.BasicDataset, lenet_quantized: bool = False)`
:   LeNet model, both in quantized and full precision version
    
    builds the model, depending on mode in either quantized or full_precision mode
    
    Args:
        lenet_quantized (bool, optional): toggles use of quantized version of lenet. Default is False.

    ### Ancestors (in MRO)

    * bitorch.models.base.Model
    * torch.nn.modules.module.Module

    ### Class variables

    `activation_function`
    :   Applies the element-wise function:
        
        .. math::
            \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
        
        Shape:
            - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
            - Output: :math:`(*)`, same shape as the input.
        
        .. image:: ../scripts/activation_images/Tanh.png
        
        Examples::
        
            >>> m = nn.Tanh()
            >>> input = torch.randn(2)
            >>> output = m(input)

    `dump_patches: bool`
    :

    `name`
    :

    `num_channels_conv`
    :

    `num_fc`
    :

    `training: bool`
    :