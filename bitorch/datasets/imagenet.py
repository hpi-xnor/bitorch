import os

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .base import BasicDataset


class ImageNetDataset(BasicDataset):
    name = "imagenet"
    num_classes = 1000
    shape = (1, 3, 224, 224)

    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        self._normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        super().__init__(*args, **kwargs)

    def get_dataset(self, train: bool, directory: str, download: bool = False) -> Dataset:
        if download and not os.path.isdir(directory):
            raise RuntimeError("ImageNet dataset must be downloaded and prepared manually.")
        if train:
            crop_scale = 0.08
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self._normalize,
            ])
            return ImageFolder(directory, transform=train_transform)
        else:
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self._normalize,
            ])
            return ImageFolder(directory, transform=test_transform)

    # TODO: we probably need to rethink the class structure :)
