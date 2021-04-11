"""Config class for quantization layers. This file should be imported before the other layers."""

from .sign import Sign
import torch


class Quantizations():
    """Class for storing quantization functions"""

    @staticmethod
    def valid_function_name(function_name: str) -> bool:
        return (
            function_name != "from_name" and
            function_name != "valid_function_name" and
            function_name in Quantizations.__dict__.keys() and
            callable(Quantizations.__dict__[function_name]))

    @staticmethod
    def from_name(function_name: str) -> function:
        assert Quantizations.valid_function_name(function_name)
        return Quantizations.__dict__[function_name]

    @staticmethod
    def default_quantization():
        return Quantizations.sign

    """
    Quantization functions
    """

    @staticmethod
    def sign(grad_cancelation_threshold=1.0):
        return Sign(grad_cancelation_threshold)

    @staticmethod
    def round():
        return torch.round


class LayerConfig():
    def __init__(self, quantization: str = "sign"):
        self.quantization = self.get_quantization_function(quantization)

    def get_quantization_function(self, quantization_name: str) -> function:
        return Quantizations.from_name(quantization_name)


global config
config = LayerConfig()
