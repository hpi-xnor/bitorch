from torch import nn


def make_initial_layers(variant: str, input_channels: int, output_channels: int) -> nn.Module:
    initial_layers = []
    if variant == "imagenet":
        initial_layers.append(nn.Conv2d(input_channels, output_channels,
                              kernel_size=7, stride=2, padding=3, bias=False))
    initial_layers.append(nn.BatchNorm2d(output_channels))
    initial_layers.append(nn.ReLU())
    initial_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return nn.Sequential(*initial_layers)
