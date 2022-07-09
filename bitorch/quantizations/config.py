from bitorch.config import Config


class QuantizationConfig(Config):
    name = "quantization_config"

    # number of bits to quantized the inputs / weights in dorefa functions
    dorefa_bits = 1

    # beta value for swishsign function
    beta = 5.0

    # scaling of progressive sign function, should be zero at the start of the training, and (close to) one at the end
    progressive_sign_scale = 0.0

    # alpha of default progressive sign transform function, should be between 2 and 10
    progressive_sign_alpha = 2

    # beta of default progressive sign transform function, should be between 2 and 10
    progressive_sign_beta = 10


config = QuantizationConfig()
