"""Config class for quantization layers. This file should be imported before the other layers."""

from .sign import Sign
import torch


class Quantizations():
    """Class for storing quantization functions"""

    @staticmethod
    def valid_function_name(function_name: str) -> bool:
        """Determines if function name is valid.

        Args:
            function_name (str): name of quanitzation function

        Returns:
            bool: True, if function is valid member function of Quantizations class, False otherwise
        """
        return (
            function_name not in ["valid_function_name", "from_name"] and
            function_name in Quantizations.__dict__.keys() and
            callable(Quantizations.__dict__[function_name]))

    @staticmethod
    def from_name(function_name: str) -> torch.nn.Module:
        """Returns the module that belongs to the function_name. This module raises an error, if the function name
        does not exist.

        Args:
            function_name (str): name of member function

        Returns:
            torch.nn.Module: Quantization Module
        """
        assert Quantizations.valid_function_name(function_name)
        return Quantizations.__dict__[function_name]

    @staticmethod
    def default_quantization() -> torch.nn.Module:
        """Returns the default quantization method"""
        return Quantizations.sign()

    """
    Quantization functions
    """

    @staticmethod
    def sign(grad_cancelation_threshold=1.0) -> torch.nn.Module:
        """Sign quantization function"""
        return Sign(grad_cancelation_threshold)


class Activations():
    """Class to store activation functions"""

    @staticmethod
    def valid_function_name(function_name: str) -> bool:
        """Determines if function name is valid.

        Args:
            function_name (str): name of activation function

        Returns:
            bool: True, if function is valid member function of Activations class, False otherwise
        """
        return (
            function_name not in ["valid_function_name", "from_name"] and
            function_name in Activations.__dict__.keys() and
            callable(Activations.__dict__[function_name]))

    @staticmethod
    def from_name(function_name: str) -> torch.nn.Module:
        """Returns the module that belongs to the function_name. This module raises an error, if the function name
        does not exist.

        Args:
            function_name (str): name of member function

        Returns:
            torch.nn.Module: Activation Module
        """
        assert Activations.valid_function_name(function_name)
        return Activations.__dict__[function_name]

    @staticmethod
    def default_activation() -> torch.nn.Module:
        """Returns the default activation method"""
        return Activations.relu()

    """
    Activation functions
    """

    @staticmethod
    def relu():
        return torch.nn.ReLU()


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
            return Quantizations.default_quantization()
        return Quantizations.from_name(quantization_name)

    def get_activation_function(self, activation_name: str = None) -> torch.nn.Module:
        """Returns the activation module specified in activation_name

        Args:
            activation_name (str, optional): name of activation function. Defaults to None.

        Returns:
            torch.nn.Module: Activation module
        """
        if activation_name is None:
            return Activations.default_activation()
        return Activations.from_name(activation_name)

    def default_activation(self):
        return Activations.default_activation()

    def default_quantization(self):
        return Quantizations.default_quantization()


# config object, global referencable
global config
config = LayerConfig()
