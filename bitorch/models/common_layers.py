from typing import List, Optional, Union
from torch import nn

from bitorch.layers.pad import PadModule


def get_initial_layers(
    variant: Optional[Union[List[int], str]], input_channels: int, output_channels: int
) -> List[nn.Module]:
    """returns the initial layers for the given variant"""
    layers: List[nn.Module] = []
    if variant == (224, 224) or variant == "imagenet":
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=7, stride=2, padding=3, bias=False))

    elif variant == "quicknet_stem":
        assert output_channels % 4 == 0
        stem_channels = output_channels // 4

        layers.append(PadModule(0, 1, 0, 1))
        layers.append(
            nn.Conv2d(
                input_channels,
                stem_channels,
                kernel_size=3,
                stride=2,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(stem_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(PadModule(0, 1, 0, 1))
        layers.append(
            nn.Conv2d(
                stem_channels,
                stem_channels,
                kernel_size=3,
                groups=stem_channels,
                stride=2,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(stem_channels, momentum=0.9))
        layers.append(
            nn.Conv2d(
                stem_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            )
        )
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
    elif variant == "grouped_stem":
        stem_width = output_channels // 2

        layers.append(nn.Conv2d(input_channels, stem_width, kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(stem_width, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, groups=4, bias=False))
        layers.append(nn.BatchNorm2d(stem_width, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(stem_width, output_channels, kernel_size=3, stride=1, padding=1, groups=8, bias=False))
    else:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))

    if variant in [(224, 224), "imagenet", "grouped_stem"]:
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return layers
