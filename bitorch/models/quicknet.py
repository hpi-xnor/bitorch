import logging
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import Module
import numpy as np

from .base import Model, NoArgparseArgsMixin
from bitorch.layers import QConv2d, PadModule
from bitorch.models.common_layers import get_initial_layers


class ResidualBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.add_module(
            "qconv",
            QConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                pad_value=1,
                padding="same",
                bias=False,
            ),
        )
        self.add_module("relu", nn.ReLU())
        self.add_module("bnorm", nn.BatchNorm2d(out_channels, momentum=0.9))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + x


class TransitionBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, strides: int) -> None:
        super().__init__()
        self.add_module("relu", nn.ReLU())
        self.add_module("pool", nn.MaxPool2d(strides, stride=1))
        self.add_module("pad", PadModule(1, 1, 1, 1))
        self.add_module(
            "depth_conv",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                groups=in_channels,
                stride=strides,
                bias=False,
            ).requires_grad_(False),
        )
        self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("relu2", nn.ReLU())
        self.add_module("norm", nn.BatchNorm2d(out_channels, momentum=0.9))


class QuickNet(Model):
    """QuickNet model from `"Larq Compute Engine: Design, Benchmark, and Deploy State-of-the-Art Binarized Neural Networks"
    <https://arxiv.org/abs/2011.09398>`_ paper.
    """

    name = "QuickNet"

    def __init__(
        self,
        input_shape: List[int],
        section_filters: Optional[List[int]] = None,
        section_blocks: Optional[List[int]] = None,
        num_classes: int = 0,
    ) -> None:
        super(QuickNet, self).__init__(input_shape, num_classes)
        if section_filters is None:
            section_filters = [64, 128, 256, 512]
        if section_blocks is None:
            section_blocks = [4, 4, 4, 4]
        self.image_channels = self._input_shape[1]
        self.num_classes = num_classes
        self.section_filters = section_filters
        self.section_blocks = section_blocks
        self._model = self._build_model()
        logging.info("building Quicknet")

        self._model.stem.apply(self._initialize_stem)  # type: ignore
        self._model.body.apply(self._initialize_body_top)  # type: ignore
        self._model.top.apply(self._initialize_body_top)  # type: ignore

    def _blurpool_init(self, weight: torch.Tensor) -> None:
        """Initialize anti-alias low_pass filter.
        See the `"Making Convolutional Networks Shift-Invariant Again" <https://arxiv.org/abs/1904.11486>`_ paper.
        """
        filters, kernel_size = weight.data.shape[0], weight.data.shape[2]

        if kernel_size == 2:
            base = np.array([1.0, 1.0])
        elif kernel_size == 3:
            base = np.array([1.0, 2.0, 1.0])
        elif kernel_size == 5:
            base = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError("filter size should be in 2, 3, 5")

        new_weights = torch.Tensor(base[:, None] * base[None, :])
        new_weights = new_weights / torch.sum(new_weights)
        new_weights = new_weights[None, None, :, :].repeat((filters, 1, 1, 1))
        weight.data = new_weights

    def _initialize_stem(self, layer: Module) -> None:
        if isinstance(layer, nn.Conv2d):
            if layer.groups == 1:
                nn.init.kaiming_normal_(layer.weight)
            else:
                nn.init.xavier_uniform_(layer.weight)

    def _initialize_body_top(self, layer: Module) -> None:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if isinstance(layer, nn.Linear) or layer.groups == 1:
                nn.init.xavier_normal_(layer.weight)
            else:
                self._blurpool_init(layer.weight)

    def _build_model(self) -> nn.Sequential:
        model = nn.Sequential()
        model.add_module(
            "stem",
            nn.Sequential(*get_initial_layers("quicknet_stem", self.image_channels, self.section_filters[0])),
        )
        body = nn.Sequential()
        for block_num, (layers, filters) in enumerate(zip(self.section_blocks, self.section_filters)):
            residual_blocks: List[Module] = []
            for layer in range(layers):
                residual_blocks.append(ResidualBlock(filters, filters))
            body.add_module(
                "ResidualBlocks_%d" % (block_num + 1),
                nn.Sequential(*residual_blocks),
            )
            if block_num != len(self.section_blocks) - 1:
                body.add_module(
                    "Transition_%d" % (block_num + 1), TransitionBlock(filters, self.section_filters[block_num + 1], 2)
                )
        model.add_module(
            "body",
            body,
        )
        model.add_module(
            "top",
            nn.Sequential(
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.section_filters[-1], self.num_classes),
            ),
        )
        return model

    def clip_weights(self, layer: Module, clip_value: float = 1.25) -> None:
        """Clips weights in quantized convolution layer in Residual Blocks"""
        if isinstance(layer, ResidualBlock):
            weights = layer.qconv.weight.data  # type: ignore
            weights = weights.clamp(-clip_value, clip_value)  # type: ignore
            layer.qconv.weight.data = weights  # type: ignore

    def on_train_batch_end(self, layer: Module) -> None:
        self.clip_weights(layer)


class QuickNetSmall(NoArgparseArgsMixin, QuickNet):
    """QuickNetSmall model from `"Larq Compute Engine: Design, Benchmark, and Deploy State-of-the-Art Binarized Neural Networks"
    <https://arxiv.org/abs/2011.09398>`_ paper.
    """

    name = "QuickNetSmall"
    section_filters = [32, 64, 256, 512]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuickNetSmall, self).__init__(section_filters=self.section_filters, *args, **kwargs)


class QuickNetLarge(NoArgparseArgsMixin, QuickNet):
    """QuickNetLarge model from `"Larq Compute Engine: Design, Benchmark, and Deploy State-of-the-Art Binarized Neural Networks"
    <https://arxiv.org/abs/2011.09398>`_ paper.
    """

    name = "QuickNetLarge"
    section_blocks = [6, 8, 12, 6]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuickNetLarge, self).__init__(section_blocks=self.section_blocks, *args, **kwargs)
