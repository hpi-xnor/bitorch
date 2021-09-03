from bitorch.config import Config


class QuantizationConfig(Config):
    name = "quantization_config"

    gradient_cancellation_threshold = 1.0
    dorefa_bits = 1


config = QuantizationConfig()
