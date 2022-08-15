from abc import ABC

from torch.utils.data import Dataset
from torchvision.datasets import cifar
from torchvision.transforms import transforms

from .base import BasicDataset

__all__ = ["CIFAR10", "CIFAR100"]


class CIFAR(BasicDataset, ABC):
    shape = (1, 3, 32, 32)
    num_train_samples = 50000
    num_val_samples = 10000

    @classmethod
    def train_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                cls.get_normalize_transform(),
            ]
        )

    @classmethod
    def test_transform(cls) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                cls.get_normalize_transform(),
            ]
        )


class CIFAR10(CIFAR):
    name = "cifar10"
    num_classes = 10
    mean = (0.4914, 0.4822, 0.4465)
    std_dev = (0.2023, 0.1994, 0.2010)

    def get_dataset(self, download: bool = True) -> Dataset:
        return cifar.CIFAR10(
            root=self.root_directory,
            train=self.is_train,
            transform=self.get_transform(),
            download=download,
        )


class CIFAR100(CIFAR):
    name = "cifar100"
    num_classes = 100
    mean = (0.507, 0.487, 0.441)
    std_dev = (0.267, 0.256, 0.276)

    def get_dataset(self, download: bool = True) -> Dataset:
        return cifar.CIFAR100(
            root=self.root_directory,
            train=self.is_train,
            transform=self.get_transform(),
            download=download,
        )
