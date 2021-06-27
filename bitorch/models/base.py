from bitorch.datasets.base import DatasetBaseClass
from torch.nn import Module
import torch
from argparse import ArgumentParser


class Model(Module):
    """Base class for Bitorch models"""
    name = "None"

    def __init__(self, dataset: DatasetBaseClass) -> None:
        super(Model, self).__init__()
        self._model = Module()
        self._dataset = dataset

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        """allows additions to the argument parser if required, e.g. to add layer count, etc.

        Args:
            parser (ArgumentParser): the argument parser
        """
        pass

    def model(self) -> Module:
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
