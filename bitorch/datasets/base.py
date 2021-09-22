import torch
from typing import Tuple
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
        """creates the dataset, has to respond to train flag, i.e. if train true the train dataset is to be created,
        if false the test dataset.

        Args:
            train (bool): toggles if train or test dataset shall be created
            directory (str): path to test/train dataset store dir (optional)
            download (bool): toggles if train/test shall be downloaded if possible

        Raises:
            NotImplementedError: thrown, because this method needs to be overwritten by subclasses

        Returns:
            Dataset: the created test/train dataset
        """
        raise NotImplementedError()

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """transforms the data tensor x. this method is called lazily every time an item is fetched.

        Args:
            x (torch.Tensor): the input tensor to be transformed

        Returns:
            torch.Tensor: the transformed tensor
        """
        return x

    def augment_dataset(self, dataset: Dataset, augmentation_level: Augmentation = Augmentation.NONE) -> Dataset:
        """custom dataset augmentation can be implemented here.

        Args:
            dataset (Dataset): the dataset to be augmented
            augmentation_level (Augmentation, optional): the augmentation level (4 different levels available).
                Defaults to Augmentation.NONE.

        Returns:
            Dataset: the augmented dataset
        """
        return dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore
        """returns the item on the specified index after applying the transform opteration.

        Args:
            index (int): requested index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the data and lable at the specified index
        """
        item = self._dataset[index]
        if isinstance(item, tuple):
            return self.transform(item[0]), item[1]

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore
