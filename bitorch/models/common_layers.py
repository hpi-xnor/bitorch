from typing import List, Optional
from torch import nn


def get_initial_layers(variant: Optional[List[int]], input_channels: int, output_channels: int) -> List[nn.Module]:
    """returns the initial layers for the given variant"""
    layers: List[nn.Module] = []
    if variant == (224, 224):  # imagenet
        layers.append(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    else:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))

    return layers
