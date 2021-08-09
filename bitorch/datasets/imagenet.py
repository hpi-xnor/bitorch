from torchvision.transforms import ToTensor, Normalize
import torch
from .base import DatasetBaseClass
from torch.utils.data import Dataset


class ImageNetDataset(DatasetBaseClass):
    name = "imagenet"
    num_classes = 1000
    shape = (1, 3, 224, 224)

    def get_dataset(self, train: bool, directory: str, download: bool = False) -> Dataset:
<<<<<<< HEAD
        return ImageNet(root=directory, train=train, transform=ToTensor(), download=download)

    # TODO: the standard transform for imagenet needed to be added here!!
=======
        raise NotImplementedError
    # TODO: the standard transform for imagenet training needed to be added
>>>>>>> d3c4ded2b893f51d0debf400bbd04b9e0bf10fd7

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
        return transform(x)
