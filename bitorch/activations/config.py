from bitorch.config import Config


class ActivationConfig(Config):
    name = "activation_config"

    # number of bits for pact activation function
    pact_bits = 1


config = ActivationConfig()
