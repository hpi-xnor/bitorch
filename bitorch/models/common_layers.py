from statistics import variance
from typing import List, Optional, Union
from torch import nn


def get_initial_layers(variant: Optional[Union[List[int], str]], input_channels: int, output_channels: int) -> List[nn.Module]:
    """returns the initial layers for the given variant"""
    layers: List[nn.Module] = []
    if variant == (224, 224) or variant == "imagenet":
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=2, padding=3, bias=False))
    elif variant == "grouped_stem":
        stem_width = output_channels // 2

        layers.append(nn.Conv2D(input_channels, stem_width, kernel_size=3, strides=2, padding=1, use_bias=False))
        layers.append(nn.BatchNorm2d(stem_width, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2D(stem_width, stem_width, kernel_size=3, strides=1, padding=1, groups=4, use_bias=False))
        layers.append(nn.BatchNorm2d(stem_width, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(
            nn.Conv2D(stem_width, stem_width * 2, kernel_size=3, strides=1, padding=1, groups=8, use_bias=False)
        )
    else:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))

    if variant in [(224, 224), "imagenet", "grouped_stem"]:
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return layers
