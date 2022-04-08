import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Any

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from bitorch.datasets.dummy_dataset import DummyDataset


class Augmentation(Enum):
    NONE = -1
    DEFAULT = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @staticmethod
    def from_string(level: str) -> "Augmentation":
        return {
            "none": Augmentation.NONE,
            "default": Augmentation.DEFAULT,
            "low": Augmentation.LOW,
            "medium": Augmentation.MEDIUM,
            "high": Augmentation.HIGH,
        }[level]


class BasicDataset(Dataset):
    name = "None"
    num_classes = 0
    shape = (0, 0, 0, 0)
    mean: Any = None
    std_dev: Any = None
    num_train_samples = 0
    num_val_samples = 0

    def __init__(
            self,
            train: bool,
            root_directory: str = None,
            download: bool = False,
            augmentation: Augmentation = Augmentation.DEFAULT) -> None:
        """initializes the dataset.

        Args:
            train (bool): whether the train or test dataset is wanted
            root_directory (str): path to main dataset storage directory
            download (bool): whether train/test should be downloaded if it does not exist
            augmentation (Augmentation): the level of augmentation (only for train dataset)

        Returns:
            Dataset: the created test/train dataset
        """
        super(BasicDataset, self).__init__()
        self.is_train = train
        self.augmentation_level = augmentation
        self._download = download
        self.root_directory = self.get_dataset_root_directory(root_directory)
        self.dataset = self.get_dataset(download)

    @classmethod
    def get_train_and_test(
            cls,
            root_directory: str,
            download: bool = False,
            augmentation: Augmentation = Augmentation.DEFAULT) -> Tuple["BasicDataset", "BasicDataset"]:
        """creates a pair of train and test dataset.

        Returns:
            Tuple: the train and test dataset
        """
        return cls(True, root_directory, download, augmentation), cls(False, root_directory, download)

    @classmethod
    def get_dummy_train_and_test_datasets(cls) -> Tuple[DummyDataset, DummyDataset]:
        train_set = DummyDataset(cls.shape, cls.num_classes, cls.num_train_samples)  # type: ignore
        val_set = DummyDataset(cls.shape, cls.num_classes, cls.num_val_samples)  # type: ignore
        return train_set, val_set

    def get_dataset_root_directory(self, root_directory_argument: Optional[str]) -> Path:
        """chooses the dataset root directory based on the passed argument or environment variables.

        Returns:
            Tuple: the train and test dataset
        """
        if root_directory_argument is not None:
            return Path(root_directory_argument)

        environment_variable_name = f"{self.name.upper()}_HOME"
        if os.environ.get(environment_variable_name) is not None:
            return Path(os.environ.get(environment_variable_name))  # type: ignore
        if os.environ.get("BITORCH_DATA_HOME") is not None:
            return Path(os.environ.get("BITORCH_DATA_HOME")) / self.name  # type: ignore

        environment_variable_hint = (
            f" To change this, set '{environment_variable_name}' or 'BITORCH_DATA_HOME' "
            f"(in the latter case, the data resides in the folder '{self.name}' in BITORCH_DATA_HOME)."
            f" Some datasets can be downloaded by adding the --download command line argument."
        )
        if self._download:
            logging.warning("Dataset is being downloaded to the directory './data'." + environment_variable_hint)
            return Path("./data")
        else:
            raise ValueError(f"Dataset {self.name} not found." + environment_variable_hint)

    def get_dataset(self, download: bool) -> Dataset:
        """creates the actual dataset

        Args:
            download (bool): toggles if train/test shall be downloaded if possible

        Raises:
            NotImplementedError: thrown, because this method needs to be overwritten by subclasses

        Returns:
            Dataset: the created test/train dataset
        """
        raise NotImplementedError()

    def get_transform(self) -> Any:
        if self.is_train:
            return self.train_transform(self.augmentation_level)
        return self.test_transform()

    @classmethod
    def test_transform(cls) -> Any:
        """get the transform for the test data.

        Returns:
            transform: the transform pipeline
        """
        return transforms.Compose([transforms.ToTensor(), cls.get_normalize_transform()])

    @classmethod
    def train_transform(cls, augmentation: Augmentation = Augmentation.DEFAULT) -> Any:
        """get the transform for the training data (should consider the current augmentation_level).

        Returns:
            transform: the transform pipeline
        """
        return transforms.Compose([transforms.ToTensor(), cls.get_normalize_transform()])

    @classmethod
    def get_normalize_transform(cls) -> transforms.Normalize:
        return transforms.Normalize(cls.mean, cls.std_dev)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """returns the item at the given index of the dataset.

        Args:
            index (int): requested index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: data and label at the specified index
        """
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def num_samples(self) -> int:
        """returns the (theoretical) dataset size."""
        return self.num_train_samples if self.is_train else self.num_val_samples
