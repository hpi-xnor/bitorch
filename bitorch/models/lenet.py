import argparse
from bitorch.layers.debug_layers import Shape_Print_Debug
from bitorch.datasets.base import DatasetBaseClass
from bitorch.layers import QLinear, QConv2d, QActivation
from torch import nn
from .base import Model


class LeNet(Model):
    """LeNet model, both in quantized and full precision version"""

    num_channels_conv = 64
    activation_function = nn.Tanh
    num_fc = 1000
    num_output = 10
    name = "lenet"

    def __init__(self, dataset: DatasetBaseClass, bits: int = 1) -> None:
        """builds the model, depending on mode in either quantized or full_precision mode

        Args:
            bits (int, optional): if bits < 32, quantized version of lenet is used, else full precision.
                Default is 1.
        """
        super(LeNet, self).__init__(dataset)
        input_channels = dataset.shape[1]
        if bits < 32:
            self._model = nn.Sequential(
                Shape_Print_Debug(name="at starts", debug_interval=1),
                nn.Conv2d(input_channels, self.num_channels_conv, kernel_size=5),
                self.activation_function(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(self.num_channels_conv),

                QConv2d(
                    self.num_channels_conv,
                    self.num_channels_conv,
                    kernel_size=5,
                    input_quantization="sign",
                    weight_quantization="round"),
                nn.BatchNorm2d(self.num_channels_conv),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),
                Shape_Print_Debug(name="after flatten", debug_interval=1),

                QActivation(activation="sign"),
                QLinear(self.num_channels_conv * 4 * 4,
                        self.num_fc, weight_quantization="sign"),
                nn.BatchNorm1d(self.num_fc),
                self.activation_function(),

                nn.Linear(self.num_fc, self.num_output),
            )
        else:
            self._model = nn.Sequential(
                nn.Conv2d(input_channels, self.num_channels_conv, kernel_size=5),
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
        parser.add_argument("--bits", type=int, choices=[1, 32], required=True,
                            help="number of bits to be used by lenet")
