"""
This submodule contains several quantization methods that can be used with our quantized layers to
build quantized models.

If you want to implement a new function, use the :code:`Quantization` base class as superclass.
"""

from typing import List, Type, Dict

from .base import Quantization
from .approx_sign import ApproxSign
from .dorefa import WeightDoReFa, InputDoReFa
from .identity import Identity
from .sign import Sign
from .ste_heaviside import SteHeaviside
from .swish_sign import SwishSign
from .progressive_sign import ProgressiveSign
from .quantization_scheduler import Quantization_Scheduler, ScheduledQuantizer
from ..util import build_lookup_dictionary

__all__ = [
    "Quantization",
    "quantization_from_name",
    "quantization_names",
    "register_custom_quantization",
    "ApproxSign",
    "InputDoReFa",
    "WeightDoReFa",
    "Identity",
    "ProgressiveSign",
    "Sign",
    "SteHeaviside",
    "SwishSign",
    "Quantization_Scheduler",
    "ScheduledQuantizer",
]


quantizations_by_name: Dict[str, Type[Quantization]] = build_lookup_dictionary(__name__, __all__, Quantization)


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


def register_custom_quantization(custom_quantization: Type[Quantization]) -> None:
    """
    Register a custom (external) quantization in bitorch.

    Args:
        custom_quantization: the custom config which should be added to bitorch
    """
    quantizations_by_name[custom_quantization.name] = custom_quantization
