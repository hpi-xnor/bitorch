from .base import DatasetBaseClass
from pathlib import Path
from typing import List
from importlib import import_module

datasets_by_name = {}

current_dir = Path(__file__).resolve().parent
for file in current_dir.iterdir():
    # grep all python files
    if file.suffix == ".py" and file.stem != "__init__":
        module = import_module(f"{__name__}.{file.stem}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            if isinstance(attr, type) and issubclass(attr, DatasetBaseClass) and attr != DatasetBaseClass:
                if attr_name in datasets_by_name:
                    raise ImportError("Two datasets found in dataset package with same name!")
                datasets_by_name[attr.name] = attr
                # make dataset accessible
                globals()[attr_name] = attr


def dataset_from_name(name: str) -> DatasetBaseClass:
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


def dataset_names() -> List:
    """getter for list of dataset names for argparse

    Returns:
        List: the dataset names
    """
    return datasets_by_name.keys()
