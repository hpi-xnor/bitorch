"""
Resnet_E implementation from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
<https://arxiv.org/abs/1906.08637>`_ paper.
"""
from .base import Model, NoArgparseArgsMixin
from typing import Optional, List, Any
import torch
import argparse
from torch import nn
import logging

from bitorch.layers import QConv2d
from bitorch.models.common_layers import get_initial_layers

__all__ = ["ResnetE34", "ResnetE18", "ResnetE"]


class BasicBlock(nn.Module):
    """BasicBlock from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    This is used for ResNetE layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing
            ResNetE uses the full-precision downsampling layer
        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=self.stride),
            nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels, momentum=0.9),
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block, i.e. two binary convolutions with batchnorms in between. Check referenced paper for
        more details.

        Returns:
            nn.Sequential: the basic building block body model
        """
        return nn.Sequential(
            QConv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels, momentum=0.9),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x through the building block.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.
        """
        residual = x
        if self.shall_downsample:
            residual = self.downsample(x)
        x = self.body(x)

        return x + residual


class SpecificResnetE(nn.Module):
    """Superclass for ResNet models"""

    def __init__(self, classes: int, channels: list) -> None:
        """builds feature and output layers

        Args:
            classes (int): number of output classes
            channels (list): the channels used in the net
        """
        super(SpecificResnetE, self).__init__()
        self.features = nn.Sequential()
        self.output_layer = nn.Linear(channels[-1], classes)

    def make_layer(self, layers: int, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        """builds a layer by stacking blocks in a sequential models.

        Args:
            layers (int): the number of blocks to stack
            in_channels (int): the input channels of this layer
            out_channels (int): the output channels of this layer
            stride (int): the stride to be used in the convolution layers

        Returns:
            nn.Sequential: the model containing the building blocks
        """
        # this tricks adds shortcut connections between original resnet blocks
        # we multiple number of blocks by 2, but add only one layer instead of two in each block
        layers = layers * 2

        layer_list: List[nn.Module] = []
        layer_list.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(layers - 1):
            layer_list.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layer_list)

    def make_feature_layers(self, layers: list, channels: list) -> List[nn.Module]:
        """builds the given layers with the specified block.

        Args:
            layers (list): the number of blocks each layer shall consist of
            channels (list): the channels

        Returns:
            nn.Sequential: [description]
        """
        feature_layers: List[nn.Module] = []
        for idx, num_layer in enumerate(layers):
            stride = 1 if idx == 0 else 2
            feature_layers.append(self.make_layer(num_layer, channels[idx], channels[idx + 1], stride))
        return feature_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor through the resnet modules

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: forwarded tensor
        """
        x = self.features(x)
        x = self.output_layer(x)
        return x


class _ResnetE(SpecificResnetE):
    """ResNetE-18 model from
    `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    """

    def __init__(
        self,
        layers: list,
        channels: list,
        classes: int,
        image_resolution: Optional[List[int]] = None,
        image_channels: int = 3,
    ) -> None:
        """Creates ResNetE model.

        Args:
            layers (list): layer sizes
            channels (list): channel num used for input/output channel size of layers. there must always be one more
                channels than there are layers.
            classes (int): number of output classes
            image_resolution (List[int], optional): resolution of input image. refer to common_layers.py.
                Defaults to None.
            image_channels (int, optional): input channels of images. Defaults to 3.

        Raises:
            ValueError: raised if the number of channels does not match number of layer + 1
        """
        super(_ResnetE, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!"
            )

        feature_layers: List[nn.Module] = []
        # feature_layers.append(nn.BatchNorm2d(image_channels, eps=2e-5, momentum=0.9))
        feature_layers.extend(get_initial_layers(image_resolution, image_channels, channels[0]))
        feature_layers.append(nn.BatchNorm2d(channels[0], momentum=0.9))

        feature_layers.extend(self.make_feature_layers(layers, channels))

        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


"""
ResNet-e specifications
"""


class ResnetE(Model):

    name = "ResnetE"

    resnet_spec = {
        18: ([2, 2, 2, 2], [64, 64, 128, 256, 512]),
        34: ([3, 4, 6, 3], [64, 64, 128, 256, 512]),
    }

    def __init__(self, resnete_num_layers: int, input_shape: List[int], num_classes: int = 0) -> None:
        super(ResnetE, self).__init__(input_shape, num_classes)
        self._model = self.create(resnete_num_layers)
        logging.info(f"building ResnetE with {str(resnete_num_layers)} layers...")

    def create(self, num_layers: int) -> nn.Module:
        """Creates a ResNetE complying to given layer number.

        Args:
            num_layers (int): number of layers to be build.

        Raises:
            ValueError: raised if no resnet specification for given num_layers is listed in the resnet_spec dict above

        Returns:
            Module: resnetE model
        """
        if num_layers not in self.resnet_spec:
            raise ValueError(f"No resnet spec for {num_layers} available!")

        layers, channels = self.resnet_spec[num_layers]
        image_channels = self._input_shape[1]
        image_resolution = self._input_shape[-2:]

        return _ResnetE(layers, channels, self._num_classes, image_resolution, image_channels)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--num-layers",
            type=int,
            choices=[18, 34],
            required=True,
            help="number of layers to be used inside resnetE",
        )


class ResnetE18(NoArgparseArgsMixin, ResnetE):
    """ResNetE-18 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    """

    name = "ResnetE18"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ResnetE18, self).__init__(18, *args, **kwargs)


class ResnetE34(NoArgparseArgsMixin, ResnetE):
    """ResNetE-34 model from `"Back to Simplicity: How to Train Accurate BNNs from Scratch?"
    <https://arxiv.org/abs/1906.08637>`_ paper.
    """

    name = "ResnetE34"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(ResnetE34, self).__init__(34, *args, **kwargs)
