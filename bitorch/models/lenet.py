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

    def __init__(self, dataset: BasicDataset, lenet_quantized: bool = False) -> None:
        """builds the model, depending on mode in either quantized or full_precision mode

        Args:
            lenet_quantized (bool, optional): toggles use of quantized version of lenet. Default is False.
        """
        super(LeNet, self).__init__(dataset)
        input_channels = dataset.shape[1]
        num_output = dataset.num_classes
        if lenet_quantized:
            self._model = nn.Sequential(
                nn.Conv2d(input_channels, self.num_channels_conv, kernel_size=5),
                self.activation_function(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(self.num_channels_conv),

                QConv2d(
                    self.num_channels_conv,
                    self.num_channels_conv,
                    kernel_size=5,
                    input_quantization="sign",
                    weight_quantization="weightdorefa"),
                nn.BatchNorm2d(self.num_channels_conv),
                nn.MaxPool2d(2, 2),
                ShapePrintDebug(),

                nn.Flatten(),

                QActivation(activation="sign"),
                QLinear(self.num_channels_conv * 4 * 4,
                        self.num_fc, weight_quantization="sign"),
                nn.BatchNorm1d(self.num_fc),
                self.activation_function(),

                nn.Linear(self.num_fc, num_output),
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

                nn.Linear(self.num_fc, num_output),
            )

    @staticmethod
    def add_argparse_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--lenet-quantized", action="store_true", default=False,
                            help="toggles use of quantized verion of lenet")
