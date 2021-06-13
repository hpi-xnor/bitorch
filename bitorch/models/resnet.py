from bitorch.layers.qconv_noact import QConv2d_NoAct
import torch
from torch import nn
from torch.nn import Module

from bitorch.layers import QConv2d
from bitorch.models.common_layers import make_initial_layers


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

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block, i.e. two binary convolutions with batchnorms in between. Check referenced paper for
        more details.

        Returns:
            nn.Sequential: the basic building block body model
        """
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
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

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> nn.Sequential:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            nn.Sequential: the downsampling model
        """
        return nn.Sequential(
            QConv2d_NoAct(self.in_channels, self.out_channels, kernel_size=1,
                          stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d_NoAct(self.in_channels, self.out_channels // 4, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(self.out_channels // 4, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(self.out_channels // 4, self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_channels)
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

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)

    def _build_downsampling(self) -> QConv2d:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            QConv2d: the downsampling convolution layer
        """
        return QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0)

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
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

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.activation = nn.ReLU()

    def _build_downsampling(self) -> QConv2d_NoAct:
        """builds the downsampling layers for rediual tensor processing

        Returns:
            QConv2d: the downsampling convolution layer
        """
        return QConv2d_NoAct(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False)

    def _build_body(self) -> nn.Sequential:
        """builds body of building block. Check referenced paper for more details.

        Returns:
            nn.Sequential: the bottleneck body model
        """
        return nn.Sequential(
            QConv2d_NoAct(self.in_channels, self.out_channels // 4, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d_NoAct(self.out_channels // 4, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
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


class ResNet(Module):
    """Superclass for ResNet models"""

    def __init__(self, classes: int, channels: list) -> None:
        """builds feature and output layers

        Args:
            classes (int): number of output classes
            channels (list): the channels used in the net
        """
        super(ResNet, self).__init__()
        self.features = nn.Sequential()
        self.output_layer = nn.Linear(channels[-1], classes)

    def make_layer(self, block: Module, layers: int, in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
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
        layer_list = []
        layer_list.append(block(in_channels, out_channels, stride))
        for _ in range(layers - 1):
            layer_list.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layer_list)

    def make_feature_layers(self, block: Module, layers: list, channels: int) -> nn.Sequential:
        """builds the given layers with the specified block.

        Args:
            block (Module): the block of which the layer shall consist
            layers (list): the number of blocks each layer shall consist of
            channels (int): the channels

        Returns:
            nn.Sequential: [description]
        """
        feature_layers = []
        for idx, num_layer in enumerate(layers):
            stride = 1 if idx == 0 else 2
            feature_layers.append(self.make_layer(block, num_layer, channels[idx], channels[idx + 1], stride))
        return nn.Sequential(*feature_layers)

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


class ResNetV1(ResNet):
    def __init__(
            self,
            block: Module,
            layers: list,
            channels: list,
            classes: int,
            initial_layers: str = "imagenet",
            image_channels: int = 3):
        super(ResNetV1, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!")

        feature_layers = []
        feature_layers.append(nn.BatchNorm2d(image_channels))
        feature_layers.append(make_initial_layers(initial_layers, image_channels, channels[0]))
        feature_layers.append(nn.BatchNorm2d(channels[0]))

        feature_layers.append(self.make_feature_layers(block, layers, channels))

        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


class ResNetV2(ResNet):
    def __init__(
            self,
            block: Module,
            layers: list,
            channels: list,
            classes: int = 1000,
            initial_layers: str = "imagenet",
            image_channels: int = 3):
        super(ResNetV2, self).__init__(classes, channels)
        if len(channels) != (len(layers) + 1):
            raise ValueError(
                f"the len of channels ({len(channels)}) must be exactly the len of layers ({len(layers)}) + 1!")

        feature_layers = []
        feature_layers.append(nn.BatchNorm2d(image_channels))
        feature_layers.append(make_initial_layers(initial_layers, image_channels, channels[0]))

        feature_layers.append(self.make_feature_layers(block, layers, channels))

        feature_layers.append(nn.BatchNorm2d(channels[-1]))
        feature_layers.append(nn.ReLU())
        feature_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_layers.append(nn.Flatten())

        self.features = nn.Sequential(*feature_layers)


"""
Resnet specifications
"""

resnet_spec = {18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
               34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
               50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}
resnet_net_versions = [ResNetV1, ResNetV2]
resnet_block_versions = [{'basic_block': BasicBlockV1, 'bottle_neck': BottleneckV1},
                         {'basic_block': BasicBlockV2, 'bottle_neck': BottleneckV2}]


def create_resnet(version, num_layers, classes: int = 1000, initial_layers: str = "imagenet", image_channels: int = 3):

    if num_layers not in resnet_spec:
        raise ValueError(f"No resnet spec for {num_layers} available!")
    if version not in [1, 2]:
        raise ValueError(f"invalid resnet version {version}, only 1 or 2 allowed")

    block_type, layers, channels = resnet_spec[num_layers]
    resnet = resnet_net_versions[version - 1]
    block = resnet_block_versions[version - 1][block_type]
    return resnet(block, layers, channels, classes, initial_layers, image_channels)


def resnet18_v1(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-18 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet18_v1 model
    """
    return create_resnet(1, 18, classes, inital_layers, image_channels)


def resnet34_v1(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-34 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet34_v1 model
    """
    return create_resnet(1, 34, classes, inital_layers, image_channels)


def resnet50_v1(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet50_v1 model
    """
    return create_resnet(1, 50, classes, inital_layers, image_channels)


def resnet152_v1(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-152 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet152_v1 model
    """
    return create_resnet(1, 152, classes, inital_layers, image_channels)


def resnet18_v2(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-18 v2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet18_v2 model
    """
    return create_resnet(2, 18, classes, inital_layers, image_channels)


def resnet34_v2(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-34 v2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet34_v2 model
    """
    return create_resnet(2, 34, classes, inital_layers, image_channels)


def resnet50_v2(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-50 v2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet50_v2 model
    """
    return create_resnet(2, 50, classes, inital_layers, image_channels)


def resnet152_v2(classes: int = 1000, inital_layers: str = "imagenet", image_channels: int = 3):
    """ResNet-152 v2 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Args:
        classes (int, optional): number of classes. Defaults to 1000.
        inital_layers (str, optional): variant of intial layers, depending on dataset. Defaults to "imagenet".
        image_channels (int, optional): color channels of input images. Defaults to 3.

    Returns:
        Module: resnet152_v2 model
    """
    return create_resnet(2, 152, classes, inital_layers, image_channels)
