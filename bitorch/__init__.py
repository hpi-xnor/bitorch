import os
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import List

from .config import Config


configs_by_name = {}

current_dir = Path(__file__).resolve().parent
files_to_iterate = list(current_dir.iterdir())
# grep all python files recursively (bfs method)
for file in files_to_iterate:
    if file.suffix == ".py" and file.stem != "__init__":

        rel_path = Path(os.path.relpath(file, current_dir))
        import_path = f"{__name__}.{str(rel_path).replace('/', '.')}"
        import_path = import_path[:import_path.rfind(".")]
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
    """getter for list of config names for argparse

    Returns:
        List: the config names
    """
    return list(configs_by_name.keys())


def add_config_args(parser: ArgumentParser) -> None:
    """adds all config arguments

    Args:
        parser (ArgumentParser): parser to add the arguments to
    """
    for config in configs_by_name.values():
        config.add_config_arguments(parser)


def apply_args_to_configuration(args: Namespace) -> None:
    """applys the cli configurations to the config objects.

    Args:
        args (Namespace): the cli configurations
    """
    for config in configs_by_name.values():
        config.apply_args_to_configuration(args)
