import argparse
import logging
from typing import Optional, List, Any

import torch
from torch import nn
from torch.nn import Module

from .densenet import BaseNetDense, DOWNSAMPLE_STRUCT, basedensenet_constructor
from .base import Model, NoArgparseArgsMixin
from bitorch.layers import QConv2d


# Blocks
class ImprovementBlock(Module):
    """ImprovementBlock which improves the last n channels"""

    def __init__(self, channels: int, in_channels: int, dilation: int = 1):
        super(ImprovementBlock, self).__init__()
        self.body_layers: List[Module] = []
        self.body_layers.append(nn.BatchNorm2d(in_channels))
        self.body_layers.append(
            QConv2d(in_channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        )

        self.use_sliced_addition = channels != in_channels
        if self.use_sliced_addition:
            assert channels < in_channels
            self.slices = [0, in_channels - channels, in_channels]
            self.slices_add_x = [False, True]
        self.body = nn.Sequential(*self.body_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.body(x)
        if not self.use_sliced_addition:
            return x + residual

        parts = []
        for add_x, slice_begin, slice_end in zip(self.slices_add_x, self.slices[:-1], self.slices[1:]):
            length = slice_end - slice_begin
            if length == 0:
                continue
            result = torch.narrow(residual, dim=1, start=slice_begin, length=length)
            if add_x:
                result = result + x
            parts.append(result)
        return torch.cat(parts, dim=1)


class _MeliusNet(BaseNetDense):
    def _add_base_block_structure(self, layer_num: int, dilation: int) -> None:
        self._add_dense_layer(layer_num, dilation)
        self.current_dense_block.add_module(
            "ImprovementBlock%d" % (layer_num + 1),
            ImprovementBlock(self.growth_rate, self.num_features, dilation=dilation),
        )


class MeliusNet(Model):
    name = "MeliusNet"

    meliusnet_spec = {
        # name: block_config,     reduction_factors,                  downsampling
        None: (None, [1 / 2, 1 / 2, 1 / 2], DOWNSAMPLE_STRUCT),
        "23": ([2, 4, 6, 6], [128 / 192, 192 / 384, 288 / 576], DOWNSAMPLE_STRUCT.replace("fp_conv", "cs,fp_conv:8")),
        "22": ([4, 5, 4, 4], [160 / 320, 224 / 480, 256 / 480], DOWNSAMPLE_STRUCT),
        "29": ([4, 6, 8, 6], [128 / 320, 192 / 512, 256 / 704], DOWNSAMPLE_STRUCT),
        "42": ([5, 8, 14, 10], [160 / 384, 256 / 672, 416 / 1152], DOWNSAMPLE_STRUCT),
        "59": ([6, 12, 24, 12], [192 / 448, 320 / 960, 544 / 1856], DOWNSAMPLE_STRUCT),
        "a": ([4, 5, 5, 6], [160 / 320, 256 / 480, 288 / 576], DOWNSAMPLE_STRUCT.replace("fp_conv", "cs,fp_conv:4")),
        "b": ([4, 6, 8, 6], [160 / 320, 224 / 544, 320 / 736], DOWNSAMPLE_STRUCT.replace("fp_conv", "cs,fp_conv:2")),
        "c": ([3, 5, 10, 6], [128 / 256, 192 / 448, 288 / 832], DOWNSAMPLE_STRUCT.replace("fp_conv", "cs,fp_conv:4")),
    }

    def __init__(
        self,
        num_layers: Optional[str],
        input_shape: List[int],
        num_classes: int = 0,
        num_init_features: int = 64,
        growth_rate: int = 64,
        bn_size: int = 0,
        dropout: float = 0,
        dilated: bool = False,
        flex_block_config: Optional[List[int]] = None,
    ) -> None:
        super(MeliusNet, self).__init__(input_shape, num_classes)
        self._model = basedensenet_constructor(
            self.meliusnet_spec,
            _MeliusNet,
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
        logging.info(f"building MeliusNet with {str(num_layers)} layers...")

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--num-layers",
            type=str,
            choices=[None, "22", "23", "29", "42", "59", "a", "b", "c"],
            required=True,
            help="number of layers to be used inside meliusnet",
        )
        parser.add_argument(
            "--reduction",
            type=str,
            required=False,
            help="divide channels by this number in transition blocks",
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


class MeliusNetFlex(MeliusNet):
    """MeliusNet-Flex model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNetFlex"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNetFlex, self).__init__(None, *args, **kwargs)

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        MeliusNet.add_argparse_arguments(parser)
        parser.add_argument(
            "--block-config",
            type=str,
            required=True,
            help="how many blocks to use in a flex model",
        )


class MeliusNet22(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-22 model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNet22"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNet22, self).__init__("22", *args, **kwargs)


class MeliusNet23(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-23 model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNet23"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNet23, self).__init__("23", *args, **kwargs)


class MeliusNet29(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-29 model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNet29"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNet29, self).__init__("29", *args, **kwargs)


class MeliusNet42(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-42 model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNet42"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNet42, self).__init__("42", *args, **kwargs)


class MeliusNet59(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-59 model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNet59"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNet59, self).__init__("59", *args, **kwargs)


class MeliusNetA(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-A model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNetA"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNetA, self).__init__("a", *args, **kwargs)


class MeliusNetB(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-B model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNetB"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNetB, self).__init__("b", *args, **kwargs)


class MeliusNetC(NoArgparseArgsMixin, MeliusNet):
    """MeliusNet-C model from `"MeliusNet: Can Binary Neural Networks Achieve MobileNet-level Accuracy?"
    <https://arxiv.org/abs/2001.05936>` paper.
    """

    name = "MeliusNetC"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(MeliusNetC, self).__init__("c", *args, **kwargs)
