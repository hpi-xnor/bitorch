from typing import List, Type

from .base import Model
from .lenet import LeNet
from .resnet import (
    Resnet,
    Resnet152_v1,
    Resnet152_v2,
    Resnet18_v1,
    Resnet18_v2,
    Resnet34_v1,
    Resnet34_v2,
    Resnet50_v1,
    Resnet50_v2,
)
from .resnet_e import (
    Resnet_E,
    Resnet_E18,
    Resnet_E34,
)
from ..util import build_lookup_dictionary

__all__ = [
    "Model", "LeNet", "Resnet", "Resnet152_v1", "Resnet152_v2", "Resnet18_v1",
    "Resnet18_v2", "Resnet34_v1", "Resnet34_v2", "Resnet50_v1", "Resnet50_v2",
    "Resnet_E", "Resnet_E18", "Resnet_E34",
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
