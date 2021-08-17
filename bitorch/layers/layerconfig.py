"""Config class for quantization layers. This file should be imported before the other layers."""

from typing import Union
from bitorch.quantizations import quantization_from_name, Quantization
import torch


class LayerConfig():
    """Class to provide layer configurations."""
    _debug_activated = True

    def get_quantization_function(self, quantization: Union[str, Quantization] = None) -> torch.nn.Module:
        """Returns the quanitization module specified in quantization_name.

        Args:
            quantization (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.

        Returns:
            torch.nn.Module: Quantization module
        """
        if quantization is None:
            return self.default_quantization()
        elif isinstance(quantization, Quantization):
            return quantization
        elif isinstance(quantization, str):
            return quantization_from_name(quantization)()
        else:
            raise ValueError(f"Invalid quantization: {quantization}")

    def default_quantization(self) -> torch.nn.Module:
        """default quantization function. used if none is passed to qlayers

        Returns:
            torch.nn.Module: default quantization module
        """
        return quantization_from_name("sign")()

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
