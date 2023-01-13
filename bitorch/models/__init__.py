"""
This submodule contains a number of adapted model architectures that use binary / quantized weights
and activations.

To define a new model, use the :code:`Model` base class as a super class.
"""

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
from .densenet import (
    DenseNet,
    DenseNet28,
    DenseNet37,
    DenseNet45,
    DenseNetFlex,
)
from .meliusnet import (
    MeliusNet,
    MeliusNet22,
    MeliusNet23,
    MeliusNet42,
    MeliusNet59,
    MeliusNetA,
    MeliusNetB,
    MeliusNetC,
    MeliusNetFlex,
)
from .resnet_e import (
    ResnetE,
    ResnetE18,
    ResnetE34,
)
from .quicknet import (
    QuickNet,
    QuickNetSmall,
    QuickNetLarge,
)
from .dlrm import DLRM
from ..util import build_lookup_dictionary

__all__ = [
    "Model",
    "model_from_name",
    "model_names",
    "register_custom_model",
    "LeNet",
    "Resnet",
    "Resnet152V1",
    "Resnet152V2",
    "Resnet18V1",
    "Resnet18V2",
    "Resnet34V1",
    "Resnet34V2",
    "Resnet50V1",
    "Resnet50V2",
    "ResnetE",
    "ResnetE18",
    "ResnetE34",
    "DLRM",
    "DenseNet",
    "DenseNet28",
    "DenseNet37",
    "DenseNet45",
    "DenseNetFlex",
    "MeliusNet",
    "MeliusNet22",
    "MeliusNet23",
    "MeliusNet42",
    "MeliusNet59",
    "MeliusNetA",
    "MeliusNetB",
    "MeliusNetC",
    "MeliusNetFlex",
    "QuickNet",
    "QuickNetSmall",
    "QuickNetLarge",
]


models_by_name = build_lookup_dictionary(__name__, __all__, Model, key_fn=lambda x: x.name.lower())


def model_from_name(name: str) -> Type[Model]:
    """
    Return a model by the given name.

    Args:
        name (str): name of the model

    Raises:
        ValueError: raised if no model under that name was found

    Returns:
        Model: the model
    """
    if name.lower() not in models_by_name:
        raise ValueError(f"{name} model not found!")
    return models_by_name[name.lower()]


def model_names() -> List[str]:
    """
    Get the list of model names.

    Returns:
        List: the model names
    """
    return list(models_by_name.keys())


def register_custom_model(custom_model: Type[Model]) -> None:
    """
    Register a custom (external) model in bitorch.

    Args:
        custom_model: the custom model which should be added to bitorch
    """
    models_by_name[custom_model.name] = custom_model
