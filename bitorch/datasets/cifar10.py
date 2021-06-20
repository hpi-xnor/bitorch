from torchvision.datasets import cifar
from torchvision.transforms import ToTensor, Normalize
import torch
from .base import DatasetBaseClass


class CIFAR10(DatasetBaseClass):
    name = "cifar10"
    num_classes = 10
    shape = (1, 3, 32, 32)

    def get_dataset(train: bool, directory: str, download: bool = True):
        return cifar.CIFAR10(root=directory, train=train, transform=ToTensor(), download=download)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        transform = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        return transform(x)
