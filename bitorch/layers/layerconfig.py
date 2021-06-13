"""Config class for quantization layers. This file should be imported before the other layers."""

from bitorch.activations.sign import Sign
from bitorch.activations.round import Round
import torch


class Quantization():
    """Class for storing quantization functions"""

    @staticmethod
    def valid_function_name(function_name: str) -> bool:
        """Determines if function name is valid.

        Args:
            function_name (str): name of quanitzation function

        Returns:
            bool: True, if function is valid member function of Quantization class, False otherwise
        """
        return (
            function_name not in ["valid_function_name", "from_name"] and
            function_name in Quantization.__dict__.keys() and
            callable(getattr(Quantization, function_name)))

    @ staticmethod
    def from_name(function_name: str) -> torch.nn.Module:
        """Returns the module that belongs to the function_name. This module raises an error, if the function name
        does not exist.

        Args:
            function_name (str): name of member function

        Raises:
            ValueError: Thrown if given function_name does not name a valid quantization function.

        Returns:
            torch.nn.Module: Quantization Module
        """
        if not Quantization.valid_function_name(function_name):
            raise ValueError(f"Quantization function name {function_name} is not valid!")
        return getattr(Quantization, function_name)()

    @ staticmethod
    def default_quantization() -> torch.nn.Module:
        """Returns the default quantization method.

        Returns:
            torch.nn.Module: the default quantization module
        """
        return Quantization.sign()

    """
    Quantization functions
    """

    @ staticmethod
    def sign(grad_cancelation_threshold: float = 1.0) -> torch.nn.Module:
        """Sign quantization function.

        Args:
            grad_cancelation_threshold (float): the threshold for gradient cancelation

        Returns:
            torch.nn.Module: sign Module
        """
        return Sign(grad_cancelation_threshold)

    @ staticmethod
    def relu() -> torch.nn.Module:
        """Rectified linear unit activation function.

        Returns:
            torch.nn.Module: relu Module
        """
        return torch.nn.ReLU()

    @ staticmethod
    def round(bits: int = 1) -> torch.nn.Module:
        """round activation function.

        Returns:
            torch.nn.Module: Round Module
        """
        return Round(bits)


class LayerConfig():
    """Class to provide layer configurations."""

    def get_quantization_function(self, quantization_name: str = None) -> torch.nn.Module:
        """Returns the quanitization module specified in quantization_name.

        Args:
            quantization_name (str, optional): name of quantization function. Defaults to None.

        Returns:
            torch.nn.Module: Quantization module
        """
        if quantization_name is None:
            return Quantization.default_quantization()
        return Quantization.from_name(quantization_name)

    def default_quantization(self) -> torch.nn.Module:
        return Quantization.default_quantization()

    def get_padding_value(self) -> float:
        return -1.


# config object, global referencable
config = LayerConfig()
