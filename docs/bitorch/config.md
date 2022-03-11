Module bitorch.config
=====================
Config class for bitorch configurations. These configs can be used to specify key default values which benefit
from beeing changed easily via argparse e.g. for training scripts.

Classes
-------

`Config()`
:   Config superclass that implements functionality to create argparse arguments for class attributes of
    subclasses.
    
    collects all attributes of class that are not the name as configurable attributes.

    ### Descendants

    * bitorch.layers.config.LayerConfig
    * bitorch.quantizations.config.QuantizationConfig

    ### Methods

    `add_config_arguments(self, parser: argparse.ArgumentParser) ‑> None`
    :   iterates over this classes configurable attributes and adds an argparse argument. in case of a boolean
        value, the value can then be toggled by either placing or leaving out the according flag.
        
        Args:
            parser (ArgumentParser): parser to add the arguments to.

    `apply_args_to_configuration(self, args: argparse.Namespace) ‑> None`
    :   loads the cli set values of configurable attributes.
        
        Args:
            args (Namespace): cli arguments