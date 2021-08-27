from typing import Tuple
import torch
from torch.utils.data import Dataset
from enum import Enum


class Augmentation(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @staticmethod
    def from_string(level: str) -> Enum:
        return {
            "none": Augmentation.NONE,
            "low": Augmentation.LOW,
            "medium": Augmentation.MEDIUM,
            "high": Augmentation.HIGH,
        }[level]


class BasicDataset(Dataset):
    name = "None"
    num_classes = 0
    shape = (0, 0, 0, 0)

    def __init__(
            self,
            train: bool,
            directory: str,
            download: bool = False,
            augmentation: Augmentation = Augmentation.NONE) -> None:
        super(BasicDataset, self).__init__()
        self._dataset = self.get_dataset(train, directory, download)
        if train:
            self._dataset = self.augment_dataset(self._dataset, augmentation)

    def get_dataset(self, train: bool, directory: str, download: bool) -> Dataset:
        return Dataset()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def augment_dataset(self, dataset: Dataset, augmentation_level: Augmentation = Augmentation.NONE) -> Dataset:
        return dataset

    def add_argparse_arguments(parser) -> None:
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        item = self._dataset[index]
        if isinstance(item, tuple):
            return self.transform(item[0]), item[1]

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore
