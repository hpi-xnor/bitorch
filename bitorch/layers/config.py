"""Config class for quantization layers. This file should be imported before the other layers."""

from typing import Union

from bitorch.config import Config
from bitorch.quantizations import quantization_from_name, Quantization


class LayerConfig(Config):
    """Class to provide layer configurations."""

    name = "layer_config"

    def get_quantization_function(self, quantization: Union[str, Quantization]) -> Quantization:
        """Returns the quantization module specified by the given name or object.

        Args:
            quantization: quantization module or name of quantization function.

        Returns:
            the quantization module
        """
        if isinstance(quantization, Quantization):
            return quantization
        elif isinstance(quantization, str):
            return quantization_from_name(quantization)()
        else:
            raise ValueError(f"Invalid quantization: {quantization}")

    # default quantization to be used in layers for inputs
    input_quantization = "sign"

    # default quantization to be used in layers for inputs
    weight_quantization = "sign"

    # toggles print / matplotlib output in debug layers
    debug_activated = False

    # default padding value used in convolution layers
    padding_value = -1.0

    # threshold used by qactivation for gradient cancellation
    gradient_cancellation_threshold = 1.0

    # bits for pact activation function
    pact_bits = 4


# config object, global referencable
config = LayerConfig()
