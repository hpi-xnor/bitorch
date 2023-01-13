from .base import Model, NoArgparseArgsMixin
from typing import Optional, List, Any
from bitorch.layers import QConv2d_NoAct
import torch
import argparse
import logging
from torch import nn
from torch.nn import Module

from bitorch.layers import QConv2d
from bitorch.models.common_layers import get_initial_layers


class BasicBlockV1(Module):
    """BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BasicBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            QConv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
            ),
            nn.BatchNorm2d(self.out_channels),
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
            ),
            nn.BatchNorm2d(self.out_channels),
            QConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
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


class BottleneckV1(Module):
    """Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BottleneckV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            QConv2d_NoAct(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                stride=self.stride,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d_NoAct(
                self.in_channels,
                self.out_channels // 4,
                kernel_size=1,
                stride=self.stride,
            ),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(
                self.out_channels // 4,
                self.out_channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(self.out_channels // 4, self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x through the building block.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.
        """
        residual = x
        x = self.body(x)
        if self.shall_downsample:
            residual = self.downsample(residual)
        x = self.activation(x + residual)
        return x


class BasicBlockV2(Module):
    """BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BasicBlockV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)

    def _build_downsampling(self) -> nn.Module:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            QConv2d: the downsampling convolution layer
        """
        return QConv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            padding=0,
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channels),
            QConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x through the building block.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.
        """
        bn = self.bn(x)
        if self.shall_downsample:
            residual = self.downsample(bn)
        else:
            residual = x
        x = self.body(bn)
        return x + residual


class BottleneckV2(Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """builds body and downsampling layers

        Args:
            in_channels (int): input channels for building block
            out_channels (int): output channels for building block
            stride (int): stride to use in convolutions
        """
        super(BottleneckV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else nn.Module()
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Module:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            QConv2d: the downsampling convolution layer
        """
        return QConv2d_NoAct(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False,
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d_NoAct(
                self.in_channels,
                self.out_channels // 4,
                kernel_size=1,
                stride=self.stride,
            ),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(
                self.out_channels // 4,
                self.out_channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(self.out_channels // 4, self.out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor x through the building block.

        Args:
            x (torch.Tensor): the input tensor

        Returns:
            torch.Tensor: the output of this building block after adding the input tensor as residual value.
        """
        residual = x
        x = self.bn(x)
        x = self.activation(x)
        if self.shall_downsample:
            residual = self.downsample(x)
        x = self.body(x)
        return x + residual


class SpecificResnet(Module):
    """Superclass for ResNet models"""

    def __init__(self, classes: int, channels: list) -> None:
        """builds feature and output layers

        Args:
            classes (int): number of output classes
            channels (list): the channels used in the net
        """
        super(SpecificResnet, self).__init__()
        self.features = nn.Sequential()
        self.output_layer = nn.Linear(channels[-1], classes)

    def make_layer(
        self,
        block: Module,
        layers: int,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        """builds a layer by stacking blocks in a sequential models.

        Args:
            block (Module): the block of which the layer shall consist
            layers (int): the number of blocks to stack
            in_channels (int): the input channels of this layer
            out_channels (int): the output channels of this layer
            stride (int): the stride to be used in the convolution layers

        Returns:
            nn.Sequential: the model containing the building blocks
        """
        layer_list: List[nn.Module] = []
        layer_list.append(block(in_channels, out_channels, stride))
        for _ in range(layers - 1):
            layer_list.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layer_list)

    def make_feature_layers(self, block: Module, layers: list, channels: list) -> List[nn.Module]:
        """builds the given layers with the specified block.

        Args:
            block (Module): the block of which the layer shall consist
            layers (list): the number of blocks each layer shall consist of
            channels (list): the channels

        Returns:
            nn.Sequential: [description]
        """
        feature_layers: List[nn.Module] = []
        for idx, num_layer in enumerate(layers):
            stride = 1 if idx == 0 else 2
            feature_layers.append(self.make_layer(block, num_layer, channels[idx], channels[idx + 1], stride))
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


class ResNetV1(SpecificResnet):
    """ResNet V1 model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    def __init__(
        self,
        block: Module,
        layers: list,
        channels: list,
        classes: int,
        image_resolution: Optional[List[int]] = None,
        image_channels: int = 3,
    ) -> None:
        """Creates ResNetV1 model.

        Args:
            block (Module): Block to be used for building the layers.
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
        super(ResNetV1, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!"
            )

        feature_layers: List[nn.Module] = []
        feature_layers.append(nn.BatchNorm2d(image_channels))
        feature_layers.extend(get_initial_layers(image_resolution, image_channels, channels[0]))
        feature_layers.append(nn.BatchNorm2d(channels[0]))

        feature_layers.extend(self.make_feature_layers(block, layers, channels))

        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


class ResNetV2(SpecificResnet):
    """ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    """

    def __init__(
        self,
        block: Module,
        layers: list,
        channels: list,
        classes: int = 1000,
        image_resolution: Optional[List[int]] = None,
        image_channels: int = 3,
    ) -> None:
        """Creates ResNetV2 model.

        Args:
            block (Module): Block to be used for building the layers.
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
        super(ResNetV2, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!"
            )

        feature_layers: List[nn.Module] = []
        feature_layers.append(nn.BatchNorm2d(image_channels))
        feature_layers.extend(get_initial_layers(image_resolution, image_channels, channels[0]))

        feature_layers.extend(self.make_feature_layers(block, layers, channels))

        feature_layers.append(nn.BatchNorm2d(channels[-1]))
        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


"""
Resnet specifications
"""


class Resnet(Model):

    name = "Resnet"

    resnet_spec = {
        18: ("basic_block", [2, 2, 2, 2], [64, 64, 128, 256, 512]),
        34: ("basic_block", [3, 4, 6, 3], [64, 64, 128, 256, 512]),
        50: ("bottle_neck", [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
        101: ("bottle_neck", [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
        152: ("bottle_neck", [3, 8, 36, 3], [64, 256, 512, 1024, 2048]),
    }
    resnet_net_versions = [ResNetV1, ResNetV2]
    resnet_block_versions = [
        {"basic_block": BasicBlockV1, "bottle_neck": BottleneckV1},
        {"basic_block": BasicBlockV2, "bottle_neck": BottleneckV2},
    ]

    def __init__(
        self,
        resnet_version: int,
        resnet_num_layers: int,
        input_shape: List[int],
        num_classes: int = 0,
    ) -> None:
        super(Resnet, self).__init__(input_shape, num_classes)
        self._model = self.create_resnet(resnet_version, resnet_num_layers)
        logging.info(f"building Resnetv{str(resnet_version)} with {str(resnet_num_layers)} layers...")

    def create_resnet(self, version: int, num_layers: int) -> Module:
        """Creates a resnet complying to given version and layer number.

        Args:
            version (int): version of resnet to be used. availavle versions are 1 or 2
            num_layers (int): number of layers to be build.

        Raises:
            ValueError: raised if no resnet specification for given num_layers is listed in the resnet_spec dict above
            ValueError: raised if invalid resnet version was passed

        Returns:
            Module: resnet model
        """
        if num_layers not in self.resnet_spec:
            raise ValueError(f"No resnet spec for {num_layers} available!")
        if version not in [1, 2]:
            raise ValueError(f"invalid resnet version {version}, only 1 or 2 allowed")

        image_channels = self._input_shape[1]
        image_resolution = self._input_shape[-2:]
        block_type, layers, channels = self.resnet_spec[num_layers]
        resnet = self.resnet_net_versions[version - 1]
        block = self.resnet_block_versions[version - 1][block_type]
        return resnet(block, layers, channels, self._num_classes, image_resolution, image_channels)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--version",
            type=int,
            choices=[1, 2],
            required=True,
            help="version of resnet to be used",
        )
        parser.add_argument(
            "--num-layers",
            type=int,
            choices=[18, 34, 50, 152],
            required=True,
            help="number of layers to be used inside resnet",
        )


class Resnet18V1(NoArgparseArgsMixin, Resnet):
    """ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet18V1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet18V1, self).__init__(1, 18, *args, **kwargs)


class Resnet34V1(NoArgparseArgsMixin, Resnet):
    """ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet34V1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet34V1, self).__init__(1, 34, *args, **kwargs)


class Resnet50V1(NoArgparseArgsMixin, Resnet):
    """ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet50V1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet50V1, self).__init__(1, 50, *args, **kwargs)


class Resnet152V1(NoArgparseArgsMixin, Resnet):
    """ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet152V1"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet152V1, self).__init__(1, 152, *args, **kwargs)


class Resnet18V2(NoArgparseArgsMixin, Resnet):
    """ResNet-18 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet18V2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet18V2, self).__init__(2, 18, *args, **kwargs)


class Resnet34V2(NoArgparseArgsMixin, Resnet):
    """ResNet-34 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet34V2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet34V2, self).__init__(2, 34, *args, **kwargs)


class Resnet50V2(NoArgparseArgsMixin, Resnet):
    """ResNet-50 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet50V2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet50V2, self).__init__(2, 50, *args, **kwargs)


class Resnet152V2(NoArgparseArgsMixin, Resnet):
    """ResNet-152 V2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    """

    name = "Resnet152V2"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Resnet152V2, self).__init__(2, 152, *args, **kwargs)
