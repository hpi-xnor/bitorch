from typing import List, Type

from .base import BasicDataset
from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet
from .mnist import MNIST
from ..util import build_lookup_dictionary

__all__ = [
    'BasicDataset', 'dataset_from_name', 'dataset_names',
    'MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet',
]


datasets_by_name = build_lookup_dictionary(__name__, __all__, BasicDataset)


def dataset_from_name(name: str) -> Type[BasicDataset]:
    """returns the dataset to which the name belongs to (name has to be the value of the datasets
    name-attribute)

    Args:
        name (str): name of the dataset

    Raises:
        ValueError: raised if no dataset under that name was found

    Returns:
        dataset: the dataset
    """
    if name not in datasets_by_name:
        raise ValueError(f"{name} dataset not found!")
    return datasets_by_name[name]


def dataset_names() -> List[str]:
    """getter for list of dataset names for argparse

    Returns:
        List: the dataset names
    """
    return list(datasets_by_name.keys())
