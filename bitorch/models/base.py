import logging
from argparse import ArgumentParser
from typing import Optional, List, Any

import torch
from torch import nn

from bitorch import RuntimeMode
from bitorch.layers import convert
from bitorch.layers.qconv1d import QConv1dBase, QConv1d_NoAct
from bitorch.layers.qconv2d import QConv2dBase, QConv2d_NoAct
from bitorch.layers.qconv3d import QConv3dBase, QConv3d_NoAct
from bitorch.models.model_hub import load_from_hub


class Model(nn.Module):
    """Base class for Bitorch models"""

    name = ""
    version_table_url = "https://api.wandb.ai/artifactsV2/default/hpi-deep-learning/QXJ0aWZhY3Q6MzE1MzQ1ODM1/a9bd2573417efc7fb8f562f06f3d322d"

    def __init__(self, input_shape: List[int], num_classes: int = 0) -> None:
        super(Model, self).__init__()
        self._model = nn.Module()
        self._input_shape = input_shape
        self._num_classes = num_classes

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        """allows additions to the argument parser if required, e.g. to add layer count, etc.

        ! please note that the inferred variable names of additional cli arguments are passed as
        keyword arguments to the constructor of this class !

        Args:
            parser (ArgumentParser): the argument parser
        """
        pass

    def model(self) -> nn.Module:
        """getter method for model

        Returns:
            Module: the main torch.nn.Module of this model
        """
        return self._model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor through the model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the model output
        """
        return self._model(x)

    def initialize(self) -> None:
        """initializes model weights a little differently for BNNs."""
        for module in self._model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # binary layers
                if isinstance(
                    module,
                    (
                        QConv1dBase,
                        QConv2dBase,
                        QConv3dBase,
                        QConv1d_NoAct,
                        QConv2d_NoAct,
                        QConv3d_NoAct,
                    ),
                ):
                    nn.init.xavier_normal_(module.weight)
                else:
                    if module.kernel_size[0] == 7:
                        # first conv layer
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    else:
                        # other 32-bit conv layers
                        nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def convert(self, new_mode: RuntimeMode, device: Optional[torch.device] = None, verbose: bool = False) -> "Model":
        return convert(self, new_mode, device, verbose)

    @classmethod
    def from_pretrained(
        cls, source: Optional[str] = None, mode: RuntimeMode = RuntimeMode.DEFAULT, **kwargs: str
    ) -> nn.Module:
        model = cls(**kwargs)  # type: ignore
        if source is not None:
            logging.info(f"Loading {cls.name} model state_dict from file {source}")
            state_dict = torch.load(source)
        else:
            kwargs["model_name"] = cls.name.lower()
            logging.info(f"Downloading {cls.name} model state_dict from hub...")
            state_dict = load_from_hub(cls.version_table_url, **kwargs)

        model.load_state_dict(state_dict)
        return model

    def on_train_batch_end(self, layer: nn.Module) -> None:
        """Is used with the pytorch lighting on_train_batch_end callback

        Implement it to e.g. clip weights after optimization. Is recursively applied to every submodule.

        Args:
            layer (nn.Module): current layer
        """
        pass


class NoArgparseArgsMixin:
    """
    Mixin for Models which subclass an existing Model, but do not have any argparse arguments anymore.

    By using this Mixin, there is no special Parser displayed for the class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        pass
