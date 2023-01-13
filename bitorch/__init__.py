"""
BITorch is a library currently under development to simplify building quantized and binary neural networks with PyTorch.
It contains implementation of the required layers, different quantization functions and examples.
"""
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import List

from .config import Config
from .runtime_mode import RuntimeMode, runtime_mode_type, change_mode, pause_wrapping  # noqa: F401
from .layers import convert  # noqa: F401

mode: RuntimeMode = RuntimeMode.DEFAULT

configs_by_name = {}

current_dir = Path(__file__).resolve().parent
files_to_iterate = list(current_dir.iterdir())
# grep all python files recursively (bfs method)
for file in files_to_iterate:
    if file.suffix == ".py" and file.stem != "__init__":

        rel_path = Path(os.path.relpath(file, current_dir))
        path_parts = list(rel_path.parent.parts) + [rel_path.stem]
        import_path = f"{__name__}.{'.'.join(path_parts)}"
        module = import_module(import_path)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # if attribute is an object of a subclass of config, store it in configs_by_name dict
            if not isinstance(attr, type) and issubclass(type(attr), Config) and type(attr) != Config:
                if attr_name in configs_by_name:
                    raise ImportError("Two configs found in config package with same name!")
                configs_by_name[attr.name] = attr
    elif file.is_dir():
        files_to_iterate += list(file.iterdir())


def config_from_name(name: str) -> Config:
    """returns the config to which the name belongs to (name has to be the value of the configs
    name-attribute)

    Args:
        name (str): name of the config

    Raises:
        ValueError: raised if no config under that name was found

    Returns:
        config: the config
    """
    if name not in configs_by_name:
        raise ValueError(f"{name} config not found!")
    return configs_by_name[name]


def config_names() -> List:
    """Get the list of config names for argparse.

    Returns:
        List: the config names
    """
    return list(configs_by_name.keys())


def add_config_args(parser: ArgumentParser) -> None:
    """Adds all arguments from all registered configs.

    Args:
        parser (ArgumentParser): parser to add the arguments to
    """
    for config_ in configs_by_name.values():
        config_.add_config_arguments(parser)


def apply_args_to_configuration(args: Namespace) -> None:
    """Applies the cli configurations to the config objects.

    Args:
        args (Namespace): the cli configurations
    """
    for config_ in configs_by_name.values():
        config_.apply_args_to_configuration(args)


def register_custom_config(custom_config: Config) -> None:
    """Register a custom (external) config in bitorch.

    Args:
        custom_config: the custom config which should be added to bitorch
    """
    configs_by_name[custom_config.name] = custom_config
