from typing import List, Type

from .base import Quantization
from .approx_sign import ApproxSign
from .dorefa import WeightDoReFa, InputDoReFa
from .identity import Identity
from .sign import Sign
from .ste_heaviside import SteHeaviside
from .swish_sign import SwishSign
from ..util import build_lookup_dictionary

__all__ = [
    "Quantization", "ApproxSign", "InputDoReFa", "WeightDoReFa", "Identity", "Sign",
    "SteHeaviside", "SwishSign",
]


quantizations_by_name = build_lookup_dictionary(__name__, __all__, Quantization)


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
