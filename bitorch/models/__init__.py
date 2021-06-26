from .base import Model
from pathlib import Path
from importlib import import_module
from typing import List

models_by_name = {}

current_dir = Path(__file__).resolve().parent
for file in current_dir.iterdir():
    # grep all python files
    if file.suffix == ".py" and file.stem != "__init__":
        module = import_module(f"{__name__}.{file.stem}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, type) and issubclass(attr, Model) and attr != Model:
                if attr_name in models_by_name:
                    raise ImportError("Two models found in model package with same name!")
                models_by_name[attr.name] = attr
                # make model accessible
                globals()[attr_name] = attr


def model_from_name(name: str) -> Model:
    """returns the model to which the name belongs to (name has to be the value of the models
    name-attribute)

    Args:
        name (str): name of the model

    Raises:
        ValueError: raised if no model under that name was found

    Returns:
        Model: the model
    """
    if name not in models_by_name:
        raise ValueError(f"{name} model not found!")
    return models_by_name[name]


def model_names() -> List:
    """getter for list of model names for argparse

    Returns:
        List: the model names
    """
    return models_by_name.keys()
