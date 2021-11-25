from typing import List
from torch import nn


def get_initial_layers(variant: str, input_channels: int, output_channels: int) -> List[nn.Module]:
    layers: List[nn.Module] = []
    if variant == "imagenet":
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif variant in ["mnist", "cifar10", "cifar100"]:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
    else:
        raise ValueError(f"Unknown initial layers for dataset '{variant}'.")

    return layers
