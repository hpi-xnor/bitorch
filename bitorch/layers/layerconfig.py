"""Config class for quantization layers. This file should be imported before the other layers."""

from bitorch.quantizations.quantization import Quantization
from bitorch.quantizations import Sign
from bitorch.quantizations import Round
import torch


class QuantizationCollection():
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
            function_name in QuantizationCollection.__dict__.keys() and
            callable(getattr(QuantizationCollection, function_name)))

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
        if not QuantizationCollection.valid_function_name(function_name):
            raise ValueError(f"Quantization function name {function_name} is not valid!")
        return getattr(QuantizationCollection, function_name)()

    @ staticmethod
    def default_quantization() -> torch.nn.Module:
        """Returns the default quantization method.

        Returns:
            torch.nn.Module: the default quantization module
        """
        return QuantizationCollection.sign()

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
    def round(bits: int = 1) -> torch.nn.Module:
        """round activation function.

        Returns:
            torch.nn.Module: Round Module
        """
        return Round(bits)


class LayerConfig():
    """Class to provide layer configurations."""
    _debug_activated = True

    def get_quantization_function(self, quantization: str = None) -> torch.nn.Module:
        """Returns the quanitization module specified in quantization_name.

        Args:
            quantization (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.

        Returns:
            torch.nn.Module: Quantization module
        """
        if quantization is None:
            return QuantizationCollection.default_quantization()
        elif isinstance(quantization, Quantization):
            return quantization
        elif isinstance(quantization, str):
            return QuantizationCollection.from_name(quantization)
        else:
            raise ValueError(f"Invalid quantization: {quantization}")

    def default_quantization(self) -> torch.nn.Module:
        """default quantization function. used if none is passed to qlayers

        Returns:
            torch.nn.Module: default quantization module
        """
        return QuantizationCollection.default_quantization()

    def get_padding_value(self) -> float:
        """default padding value used in qconvolution layers (neccessary because 0 padding does not make much sense in a
        binary network which only uses -1 and 1 as values)

        Returns:
            float: default padding value
        """
        return -1.

    def debug_activated(self) -> bool:
        """function to get the current debug activation status. Debug layers won't output if this function returns False.

        Returns:
            bool: debug activation flag
        """
        return self._debug_activated

    def activate_debug(self, debug: bool) -> None:
        """setter method for debug mode.

        Args:
            debug (bool): flag that determines wether there should be debug output from debug layers
        """
        self._debug_activated = debug


# config object, global referencable
config = LayerConfig()
