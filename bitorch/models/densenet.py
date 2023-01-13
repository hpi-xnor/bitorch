import logging
import argparse
from typing import Any, List, Optional, Type, Union

import torch
from torch import nn
from torch.nn import Module, ChannelShuffle

from .base import Model, NoArgparseArgsMixin
from bitorch.layers import QConv2d
from bitorch.models.common_layers import get_initial_layers


class DenseLayer(Module):
    def __init__(self, num_features: int, growth_rate: int, bn_size: int, dilation: int, dropout: float):
        super(DenseLayer, self).__init__()
        self.dropout = dropout
        self.num_features = num_features
        self.feature_list: List[Module] = []
        if bn_size == 0:
            # no bottleneck
            self._add_conv_block(
                QConv2d(self.num_features, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
            )
        else:
            self._add_conv_block(QConv2d(self.num_features, bn_size * growth_rate, kernel_size=1))
            self._add_conv_block(QConv2d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1))
        self.features = nn.Sequential(*self.feature_list)

    def _add_conv_block(self, layer: Module) -> None:
        self.feature_list.append(nn.BatchNorm2d(self.num_features))
        self.feature_list.append(layer)
        if self.dropout:
            self.feature_list.append(nn.Dropout(self.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ident = x
        x = self.features(x)
        x = torch.cat([ident, x], dim=1)
        return x


class BaseNetDense(Module):
    """Densenet-BC model from the
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_ paper.
    """

    def __init__(
        self,
        num_init_features: int,
        growth_rate: int,
        block_config: List[int],
        reduction: List[float],
        bn_size: int,
        downsample: str,
        image_resolution: Optional[List[int]] = None,
        dropout: float = 0,
        classes: int = 1000,
        image_channels: int = 3,
        dilated: bool = False,
    ):
        super(BaseNetDense, self).__init__()
        self.num_blocks = len(block_config)
        self.dilation = (1, 1, 2, 4) if dilated else (1, 1, 1, 1)
        self.downsample_struct = downsample
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.dropout = dropout
        self.reduction_rates = reduction
        self.num_features = num_init_features

        self.features = nn.Sequential(*get_initial_layers(image_resolution, image_channels, self.num_features))
        # Add dense blocks
        for i, repeat_num in enumerate(block_config):
            self._make_repeated_base_blocks(repeat_num, i)
            if i != len(block_config) - 1:
                self._make_transition(i)
        self.finalize = nn.Sequential(
            nn.BatchNorm2d(self.num_features), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.output = nn.Linear(self.num_features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.finalize(x)
        x = self.output(x)
        return x

    def _add_base_block_structure(self, layer_num: int, dilation: int) -> None:
        raise NotImplementedError()

    def _make_repeated_base_blocks(self, num_base_blocks: int, stage_index: int) -> None:
        dilation = self.dilation[stage_index]
        self.current_dense_block = nn.Sequential()
        for i in range(num_base_blocks):
            self._add_base_block_structure(i, dilation)
        self.features.add_module("DenseBlock_%d" % (stage_index + 1), self.current_dense_block)

    def _add_dense_layer(self, layer_num: int, dilation: int) -> None:
        dense_layer = DenseLayer(self.num_features, self.growth_rate, self.bn_size, dilation, self.dropout)
        self.num_features += self.growth_rate
        self.current_dense_block.add_module("DenseLayer_%d" % (layer_num + 1), dense_layer)

    def _make_transition(self, transition_num: int) -> None:
        dilation = self.dilation[transition_num + 1]
        num_out_features = self.num_features // self.reduction_rates[transition_num]
        num_out_features = int(round(num_out_features / 32)) * 32

        transition_layers: List[Module] = []

        for layer in self.downsample_struct.split(","):
            if layer == "bn":
                transition_layers.append(nn.BatchNorm2d(self.num_features))
            elif layer == "relu":
                transition_layers.append(nn.ReLU())
            elif layer == "q_conv":
                transition_layers.append(QConv2d(self.num_features, num_out_features, kernel_size=1))
            elif "fp_conv" in layer:
                groups = 1
                if ":" in layer:
                    groups = int(layer.split(":")[1])
                transition_layers.append(
                    nn.Conv2d(self.num_features, num_out_features, kernel_size=1, groups=groups, bias=False)
                )
            elif layer == "pool" and dilation == 1:
                transition_layers.append(nn.AvgPool2d(2, stride=2))
            elif layer == "max_pool" and dilation == 1:
                transition_layers.append(nn.MaxPool2d(2, stride=2))
            elif "cs" in layer:
                groups = 16
                if ":" in layer:
                    groups = int(layer.split(":")[1])
                transition_layers.append(ChannelShuffle(groups))

        transition = nn.Sequential(*transition_layers)

        self.features.add_module("Transition_%d" % (transition_num + 1), transition)
        self.num_features = num_out_features


class _DenseNet(BaseNetDense):
    def _add_base_block_structure(self, layer_num: int, dilation: int) -> None:
        self._add_dense_layer(layer_num, dilation)


def basedensenet_constructor(
    spec: dict,
    model: Type[BaseNetDense],
    num_layers: Optional[Union[int, str]],
    num_init_features: int,
    growth_rate: int,
    bn_size: int,
    dropout: float,
    dilated: bool,
    flex_block_config: Optional[List[int]],
    classes: int = 1000,
    image_resolution: Optional[List[int]] = None,
    image_channels: int = 3,
) -> Module:
    """Creates a densenet of the given model type with given layer numbers.

    Args:
        spec (dict): specification that holds block config, reduction factors and downsample layer names
        model (Type[BaseNetDense]): the model to instantiate.
        num_layers (int): number of layers to be build.
        num_init_features (int, optional): number of initial features.
        growth_rate (int, optional): growth rate of the channels.
        bn_size (int, optional): size of the bottleneck.
        dropout (float, optional): dropout percentage in dense layers.
        dilated (bool, optional): whether to use dilation in convolutions.
        flex_block_config (List[int], optional) number of blocks in a flex model.
        classes (int, optional): number of output classes. Defaults to 1000.
        image_resolution (List[int], optional): determines set of initial layers to be used. Defaults to None.
        image_channels (int, optional): number of channels of input images. Defaults to 3.

    Raises:
        ValueError: raised if no specification for given num_layers is listed in the given spec dict,
                    block config is not given as a list of ints,
                    number of reductions is incorrect

    Returns:
        Module: instance of model
    """
    if num_layers not in spec:
        raise ValueError(f"No spec for {num_layers} available!")

    block_config, reduction_factor, downsampling = spec[num_layers]

    if num_layers is None and flex_block_config is not None:
        block_config = flex_block_config

    reduction = [1 / x for x in reduction_factor]
    if not isinstance(block_config, List):
        raise ValueError(f"block config {block_config} must be a list")
    if not len(reduction) == len(block_config) - 1:
        raise ValueError(f'"wrong number of reductions, should be {len(block_config) - 1}"')

    return model(
        num_init_features,
        growth_rate,
        block_config,
        reduction,
        bn_size,
        downsampling,
        image_resolution,
        dropout,
        classes,
        image_channels,
        dilated,
    )


"""
DenseNet specifications
"""

DOWNSAMPLE_STRUCT = "bn,max_pool,relu,fp_conv"


class DenseNet(Model):
    name = "DenseNet"
    densenet_spec = {
        # block_config, reduction_factor, downsampling
        None: (None, [1 / 2, 1 / 2, 1 / 2], DOWNSAMPLE_STRUCT),
        28: ([6, 6, 6, 5], [1 / 2.7, 1 / 2.7, 1 / 2.2], DOWNSAMPLE_STRUCT),
        37: ([6, 8, 12, 6], [1 / 3.3, 1 / 3.3, 1 / 4], DOWNSAMPLE_STRUCT),
        45: ([6, 12, 14, 8], [1 / 2.7, 1 / 3.3, 1 / 4], DOWNSAMPLE_STRUCT),
    }

    def __init__(
        self,
        num_layers: Optional[int],
        input_shape: List[int],
        num_classes: int = 0,
        num_init_features: int = 64,
        growth_rate: int = 64,
        bn_size: int = 0,
        dropout: float = 0,
        dilated: bool = False,
        flex_block_config: Optional[List[int]] = None,
    ) -> None:
        super(DenseNet, self).__init__(input_shape, num_classes)
        self._model = basedensenet_constructor(
            self.densenet_spec,
            _DenseNet,
            num_layers,
            num_init_features,
            growth_rate,
            bn_size,
            dropout,
            dilated,
            flex_block_config,
            self._num_classes,
            self._input_shape[-2:],
            self._input_shape[1],
        )
        logging.info(f"building DenseNet with {str(num_layers)} layers...")

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--num-layers",
            type=int,
            choices=[None, 28, 37, 45],
            required=True,
            help="number of layers to be used inside densenet",
        )
        parser.add_argument(
            "--reduction",
            type=str,
            required=False,
            help='divide channels by this number in transition blocks (3 values, e.g. "2,2.5,3")',
        )
        parser.add_argument(
            "--growth-rate",
            type=int,
            required=False,
            help="add this many features each block",
        )
        parser.add_argument(
            "--init-features",
            type=int,
            required=False,
            help="start with this many filters in the first layer",
        )
        parser.add_argument(
            "--downsample-structure",
            type=str,
            required=False,
            help="layers in downsampling branch (available: bn,relu,conv,fp_conv,pool,max_pool)",
        )


class DenseNetFlex(DenseNet):
    """
    Flexible BinaryDenseNet model from `"BinaryDenseNet: Developing an Architecture for Binary Neural Networks"
    <https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Bethge_BinaryDenseNet_Developing_an_Architecture_for_Binary_Neural_Networks_ICCVW_2019_paper.html>` paper.
    """

    name = "DenseNetFlex"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(DenseNetFlex, self).__init__(None, *args, **kwargs)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        DenseNet.add_argparse_arguments(parser)
        parser.add_argument(
            "--block-config",
            type=str,
            required=True,
            help="how many blocks to use in a flex model",
        )


class DenseNet28(NoArgparseArgsMixin, DenseNet):
    """
    BinaryDenseNet-28 model from `"BinaryDenseNet: Developing an Architecture for Binary Neural Networks"` paper.

    .. _"BinaryDenseNet: Developing an Architecture for Binary Neural Networks":
    https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Bethge_BinaryDenseNet_Developing_an_Architecture_for_Binary_Neural_Networks_ICCVW_2019_paper.html
    """

    name = "DenseNet28"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(DenseNet28, self).__init__(28, *args, **kwargs)


class DenseNet37(NoArgparseArgsMixin, DenseNet):
    """
    BinaryDenseNet-37 model from `"BinaryDenseNet: Developing an Architecture for Binary Neural Networks"` paper.

    .. _"BinaryDenseNet: Developing an Architecture for Binary Neural Networks":
    https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Bethge_BinaryDenseNet_Developing_an_Architecture_for_Binary_Neural_Networks_ICCVW_2019_paper.html
    """

    name = "DenseNet37"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(DenseNet37, self).__init__(37, *args, **kwargs)


class DenseNet45(NoArgparseArgsMixin, DenseNet):
    """
    BinaryDenseNet-45 model from `"BinaryDenseNet: Developing an Architecture for Binary Neural Networks"` paper.

    .. _"BinaryDenseNet: Developing an Architecture for Binary Neural Networks":
    https://openaccess.thecvf.com/content_ICCVW_2019/html/NeurArch/Bethge_BinaryDenseNet_Developing_an_Architecture_for_Binary_Neural_Networks_ICCVW_2019_paper.html
    """

    name = "DenseNet45"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(DenseNet45, self).__init__(45, *args, **kwargs)
