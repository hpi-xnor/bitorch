from .base import Quantization
from pathlib import Path
from typing import List, Type
from importlib import import_module

quantizations_by_name = {}

current_dir = Path(__file__).resolve().parent
for file in current_dir.iterdir():
    # grep all python files
    if file.suffix == ".py" and file.stem != "__init__":
        module = import_module(f"{__name__}.{file.stem}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, type) and issubclass(attr, Quantization) and attr != Quantization:
                if attr_name in quantizations_by_name:
                    raise ImportError("Two quantizations found in quantization package with same name!")
                quantizations_by_name[attr.name] = attr
                # make quantization accessible
                globals()[attr_name] = attr


def quantization_from_name(name: str) -> Type[Quantization]:
    """returns the quantization to which the name belongs to (name has to be the value of the quantizations
    name-attribute)

    Args:
        name (str): name of the quantization

    Raises:
        ValueError: raised if no quantization under that name was found

    Returns:
        quantization: the quantization
    """
    if name not in quantizations_by_name:
        raise ValueError(f"{name} quantization not found!")
    return quantizations_by_name[name]


def quantization_names() -> List:
    """getter for list of quantization names for argparse

    Returns:
        List: the quantization names
    """
    return list(quantizations_by_name.keys())
