Module bitorch.layers.config
============================
Config class for quantization layers. This file should be imported before the other layers.

Classes
-------

`LayerConfig()`
:   Class to provide layer configurations.
    
    collects all attributes of class that are not the name as configurable attributes.

    ### Ancestors (in MRO)

    * bitorch.config.Config

    ### Class variables

    `debug_activated`
    :

    `gradient_cancellation_threshold`
    :

    `input_quantization`
    :

    `name`
    :

    `padding_value`
    :

    `weight_quantization`
    :

    ### Methods

    `get_quantization_function(self, quantization: Union[str, bitorch.quantizations.base.Quantization]) ‑> torch.nn.modules.module.Module`
    :   Returns the quanitization module specified in quantization_name.
        
        Args:
            quantization (Union[str, Quantization]): quantization module or name of quantization function.
        
        Returns:
            torch.nn.Module: Quantization module