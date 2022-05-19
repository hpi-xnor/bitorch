import argparse
from bitorch.layers.debug_layers import ShapePrintDebug
from bitorch.datasets.base import BasicDataset
from bitorch.layers import QLinear, QConv2d, QActivation
from torch import nn
from .base import Model


class LeNet(Model):
    """LeNet model, both in quantized and full precision version"""

    num_channels_conv = 64
    activation_function = nn.Tanh
    num_fc = 1000
    name = "lenet"

    def generate_quant_model(self, weight_quant: str, input_quant: str,
                             weight_quant_2: str = None, input_quant_2: str = None) -> nn.Sequential:
        weight_quant_2 = weight_quant_2 or weight_quant
        input_quant_2 = input_quant_2 or input_quant

        model = nn.Sequential(
            nn.Conv2d(self.input_channels, self.num_channels_conv, kernel_size=5),
            self.activation_function(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.num_channels_conv),

            QConv2d(
                self.num_channels_conv,
                self.num_channels_conv,
                kernel_size=5,
                input_quantization=input_quant,
                weight_quantization=weight_quant),
            nn.BatchNorm2d(self.num_channels_conv),
            nn.MaxPool2d(2, 2),
            ShapePrintDebug(),

            nn.Flatten(),

            QActivation(activation=input_quant_2),
            QLinear(self.num_channels_conv * 4 * 4,
                    self.num_fc, weight_quantization=weight_quant_2),
            nn.BatchNorm1d(self.num_fc),
            self.activation_function(),

            nn.Linear(self.num_fc, self.num_output),
        )
        return model

    def __init__(self, dataset: BasicDataset, lenet_version: int = 0) -> None:
        """builds the model, depending on mode in either quantized or full_precision mode

        Args:
            lenet_quantized (bool, optional): toggles use of quantized version of lenet. Default is False.
        """
        super(LeNet, self).__init__(dataset)
        self.input_channels = dataset.shape[1]
        self.num_output = dataset.num_classes
        if lenet_version == 0:
            self._model = self.generate_quant_model("sign", "sign")
        elif lenet_version == 1:
            self._model = self.generate_quant_model("weightdorefa", "weightdorefa")
        elif lenet_version == 2:
            self._model = self.generate_quant_model("sign", "weightdorefa", weight_quant_2="sign")
        elif lenet_version == 3:
            self._model = self.generate_quant_model("sign", "weightdorefa")
        else:
            self._model = nn.Sequential(
                nn.Conv2d(self.input_channels, self.num_channels_conv, kernel_size=5),
                nn.BatchNorm2d(self.num_channels_conv),
                self.activation_function(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(self.num_channels_conv, self.num_channels_conv, kernel_size=5),
                nn.BatchNorm2d(self.num_channels_conv),
                self.activation_function(),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),

                nn.Linear(self.num_channels_conv * 4 * 4, self.num_fc),
                nn.BatchNorm1d(self.num_fc),
                self.activation_function(),

                nn.Linear(self.num_fc, self.num_output),
            )

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lenet-version", type=int, default=0,
                            help="choses a verion of lenet")
