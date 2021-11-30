from bitorch.config import Config


class QuantizationConfig(Config):
    name = "quantization_config"

    # number of bits to quantized the inputs / weights in dorefa functions
    dorefa_bits = 1

    # beta value for swishsign function
    beta = 5.0


config = QuantizationConfig()
