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
from bitorch.util import is_url


class Model(nn.Module):
    """Base class for Bitorch models"""

    name = ""
    pretrained_model_url = None

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

    def _generate_checkpoint_name(self):
        # TODO: encode current runtime / layer implementation in name for better reference / correct loading of model
        return f"{self.name}_checkpoint.pth"

    def from_pretrained(self, source: Optional[str] = None, verbose: bool = False) -> nn.Module:
        if source is not None:
            return self._load_from_source(source, verbose)
        elif self.pretrained_model_url is not None:
            return self._load_from_url(self.pretrained_model_url)
        raise ValueError(
            f"Model {self.name} does not have a predefined download url, "
            "you have to specify a source to load the model from the specified checkpoint!"
        )

    def _load_from_url(self, url: str, verbose: bool = False):
        ...

    def _load_from_source(self, source: str, verbose: bool = False):
        ...

    def _upload_model(self, source_path: str, destination_url: str):
        ...

    def store_checkpoint(self, destination: Optional[str] = None) -> None:
        checkpoint_name = self._generate_checkpoint_name()
        if is_url(destination):
            logging.info(f"uploading model to url: {destination}...")
            tmp_path = f"/tmp/{checkpoint_name}"
            torch.save(self, tmp_path)
            self._upload_model(tmp_path, destination)
        else:
            logging.debug(f"saving model to {destination}...")
            torch.save(self, destination)
        logging.debug("done saving model!")


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
