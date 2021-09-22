from bitorch.datasets.base import BasicDataset
from torch.nn import Module
from typing import Union, Type
import torch
from argparse import ArgumentParser


class Model(Module):
    """Base class for Bitorch models"""
    name = "None"

    def __init__(self, dataset: Union[BasicDataset, Type[BasicDataset]]) -> None:
        super(Model, self).__init__()
        self._model = Module()
        self._dataset = dataset

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        """allows additions to the argument parser if required, e.g. to add layer count, etc.

        ! please note that the infered variable names of additional cli arguments are passed as
        keyword arguments to the constructor of this class !

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
