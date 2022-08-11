from typing import List
from torch import nn


def get_initial_layers(variant: str, input_channels: int, output_channels: int) -> List[nn.Module]:
    """Get commonly used layers to extract initial features from the image."""
    layers: List[nn.Module] = []
    if variant == "imagenet":
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
    elif variant in ["mnist", "cifar10", "cifar100"]:
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False))
    else:
        raise ValueError(f"Unknown initial layers for dataset '{variant}'.")

    if variant in ["imagenet", "grouped_stem"]:
        layers.append(nn.BatchNorm2d(output_channels, momentum=0.9))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    return layers
