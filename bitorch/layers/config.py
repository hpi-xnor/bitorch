"""Config class for quantization layers. This file should be imported before the other layers."""

from typing import Union
import torch

from bitorch.config import Config
from bitorch.quantizations import quantization_from_name, Quantization


class LayerConfig(Config):
    """Class to provide layer configurations."""

    name = "layer_config"

    def get_quantization_function(self, quantization: Union[str, Quantization] = None) -> torch.nn.Module:
        """Returns the quanitization module specified in quantization_name.

        Args:
            quantization (Union[str, Quantization], optional): quantization module or name of quantization function.
                Defaults to None.

        Returns:
            torch.nn.Module: Quantization module
        """
        if quantization is None:
            return self.quantization()
        elif isinstance(quantization, Quantization):
            return quantization
        elif isinstance(quantization, str):
            return quantization_from_name(quantization)()
        else:
            raise ValueError(f"Invalid quantization: {quantization}")

    # default quantization to be used in layers
    quantization = quantization_from_name("sign")

    # toggles print / matplotlib output in debug layers
    debug_activated = False

    # default padding value used in convolution layers
    padding_value = -1.0


# config object, global referencable
config = LayerConfig()
