from typing import List, Type

from .base import Model
from .lenet import LeNet
from .resnet import (
    Resnet,
    Resnet152V1,
    Resnet152V2,
    Resnet18V1,
    Resnet18V2,
    Resnet34V1,
    Resnet34V2,
    Resnet50V1,
    Resnet50V2,
)
from .resnet_e import (
    ResnetE,
    ResnetE18,
    ResnetE34,
)
from ..util import build_lookup_dictionary

__all__ = [
    "Model", "LeNet", "Resnet", "Resnet152V1", "Resnet152V2", "Resnet18V1",
    "Resnet18V2", "Resnet34V1", "Resnet34V2", "Resnet50V1", "Resnet50V2",
    "ResnetE", "ResnetE18", "ResnetE34",
]


models_by_name = build_lookup_dictionary(__name__, __all__, Model)


def model_from_name(name: str) -> Type[Model]:
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
    return list(models_by_name.keys())
