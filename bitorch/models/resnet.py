import torch
from torch import nn
from torch.nn import Module

from bitorch.layers import QConv2d
from bitorch.models.common_layers import make_initial_layers


class BasicBlockV1(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlockV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.out_channels),
            QConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.body(x)
        if self.shall_downsample:
            residual = self.downsample(residual)
        x = self.activation(x + residual)
        return x


class BottleneckV1(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckV1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.activation = nn.ReLU()

    def _build_downsampling(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels),
        )

    def _build_body(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels // 4, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d(self.out_channels // 4, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d(self.out_channels // 4, self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_channels)
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.body(x)
        if self.shall_downsample:
            residual = self.downsample(residual)
        x = self.activation(x + residual)
        return x


class BasicBlockV2(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlockV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)

    def _build_downsampling(self):
        return QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, padding=0)

    def _build_body(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.out_channels),
            QConv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor):
        bn = self.bn(x)
        if self.shall_downsample:
            residual = self.downsample(bn)
        else:
            residual = x
        x = self.body(bn)
        return x + residual


class BottleneckV2(Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckV2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.shall_downsample = self.in_channels != self.out_channels

        self.downsample = self._build_downsampling() if self.shall_downsample else None
        self.body = self._build_body()
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.activation = nn.ReLU()

    def _build_downsampling(self):
        return QConv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.stride, bias=False)

    def _build_body(self):
        return nn.Sequential(
            QConv2d(self.in_channels, self.out_channels // 4, kernel_size=1, stride=self.stride),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d(self.out_channels // 4, self.out_channels // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_channels // 4),
            nn.ReLU(),
            QConv2d(self.out_channels // 4, self.out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.bn(x)
        x = self.activation(x)
        if self.shall_downsample:
            residual = self.downsample(x)
        x = self.body(x)
        return x + residual


class ResNet(Module):
    def __init__(self, classes, channels):
        super(ResNet, self).__init__()
        self.features = nn.Sequential()
        self.output_layer = nn.Linear(channels[-1], classes)

    def make_layer(self, block, layers, in_channels, out_channels, stride):
        layer_list = []
        layer_list.append(block(in_channels, out_channels, stride))
        for _ in range(layers - 1):
            layer_list.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layer_list)

    def make_feature_layers(self, block, layers, channels):
        feature_layers = []
        for idx, num_layer in enumerate(layers):
            stride = 1 if idx == 0 else 2
            feature_layers.append(self.make_layer(block, num_layer, channels[idx], channels[idx + 1], stride))
        return nn.Sequential(*feature_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
