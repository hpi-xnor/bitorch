import torch
from torch.utils.data import Dataset
from enum import Enum


class Augmentation(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3

    @staticmethod
    def from_string(level: str) -> int:
        return {
            "none": Augmentation.NONE,
            "low": Augmentation.LOW,
            "medium": Augmentation.MEDIUM,
            "high": Augmentation.HIGH,
        }[level]


class DatasetBaseClass(Dataset):
    name = "None"
    num_classes = 0
    shape = ()

    def __init__(
            self,
            train: bool,
            directory: str,
            download: bool = False,
            augmentation: int = Augmentation.NONE) -> None:
        super(DatasetBaseClass, self).__init__()
        self._dataset = self.get_dataset(train, directory, download)
        if train:
            self._dataset = self.augment_dataset(self._dataset, augmentation)

    def get_dataset(self, train: bool, directory: str, download: bool) -> Dataset:
        raise NotImplementedError()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def augment_dataset(self, dataset: Dataset, augmentation_level: int = Augmentation.NONE):
        return dataset

    def add_argparse_arguments(parser):
        pass

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.transform(self._dataset[index])

    def __len__(self):
        return len(self._dataset)
