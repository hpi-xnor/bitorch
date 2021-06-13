from torch import nn
from bitorch.layers import Shape_Print_Debug_Layer


def make_initial_layers(variant: str, input_channels: int, output_channels: int) -> nn.Module:
    initial_layers = []
    if variant == "imagenet":
        initial_layers.append(nn.Conv2d(input_channels, output_channels,
                              kernel_size=7, stride=2, padding=3, bias=False))
    elif variant == "mnist":
        initial_layers.append(nn.Conv2d(input_channels, output_channels,
                                        kernel_size=(7, 7),
                                        stride=(2, 2),
                                        padding=(3, 3), bias=False))
        # initial_layers.append(nn.Conv2d(input_channels, output_channels,
        #                       kernel_size=3, stride=1, padding=1, bias=False))
    initial_layers.append(nn.BatchNorm2d(output_channels))
    initial_layers.append(nn.ReLU())
    initial_layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    return nn.Sequential(*initial_layers)
