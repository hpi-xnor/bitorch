from torchvision.datasets import mnist
from torchvision.transforms import ToTensor
from .base import DatasetBaseClass


class MNIST(DatasetBaseClass):
    name = "mnist"
    num_classes = 10
    shape = (1, 1, 28, 28)

    def get_dataset(self, train: bool, directory: str, download: bool = True):
        return mnist.MNIST(root=directory, train=train, transform=ToTensor(), download=download)
