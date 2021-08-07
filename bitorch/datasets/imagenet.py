from torchvision.transforms import ToTensor, Normalize
import torch
from .base import DatasetBaseClass
from torch.utils.data import Dataset


class ImageNet(DatasetBaseClass):
    name = "imagenet"
    num_classes = 1000
    shape = (1, 3, 224, 224)

    def get_dataset(self, train: bool, directory: str, download: bool = False) -> Dataset:
        raise NotImplementedError
    # TODO: the standard transform for imagenet training needed to be added
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        transform = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
        return transform(x)
