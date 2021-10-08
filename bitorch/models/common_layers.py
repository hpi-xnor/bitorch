from typing import List
from torch import nn


def make_initial_layers(variant: str, input_channels: int, output_channels: int) -> nn.Module:
    initial_layers: List[nn.Module] = []
    if variant == "imagenet":
        initial_layers.append(nn.Conv2d(input_channels, output_channels,
                              kernel_size=7, stride=2, padding=3, bias=False))
        initial_layers.append(nn.BatchNorm2d(output_channels))
        initial_layers.append(nn.ReLU())
        initial_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    elif variant in ["mnist", "cifar10", "cifar100"]:
        initial_layers.append(nn.Conv2d(input_channels, output_channels,
                                        kernel_size=(3, 3),
                                        padding=(1, 1), bias=False))
    else:
        raise ValueError(f"Unknown initial layers for dataset '{variant}'.")

    return nn.Sequential(*initial_layers)
