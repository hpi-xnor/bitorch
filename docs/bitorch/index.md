Module bitorch
==============

Sub-modules
-----------
* bitorch.config
* bitorch.datasets
* bitorch.layers
* bitorch.models
* bitorch.optimization
* bitorch.quantizations
* bitorch.util

Functions
---------

    
`add_config_args(parser: argparse.ArgumentParser) ‑> None`
:   adds all config arguments
    
    Args:
        parser (ArgumentParser): parser to add the arguments to

    
`apply_args_to_configuration(args: argparse.Namespace) ‑> None`
:   applys the cli configurations to the config objects.
    
    Args:
        args (Namespace): the cli configurations

    
`config_from_name(name: str) ‑> bitorch.config.Config`
:   returns the config to which the name belongs to (name has to be the value of the configs
    name-attribute)
    
    Args:
        name (str): name of the config
    
    Raises:
        ValueError: raised if no config under that name was found
    
    Returns:
        config: the config

    
`config_names() ‑> List`
:   getter for list of config names for argparse
    
    Returns:
        List: the config names