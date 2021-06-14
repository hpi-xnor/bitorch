from bitorch.layers.qlinear import QLinear
from bitorch.layers.qconv import QConv2d
from bitorch.layers.qactivation import QActivation
from torch import nn, Tensor


class LeNet(nn.Module):
    """LeNet model, both in quantized and full precision version"""

    num_channels_conv = 64
    activation_function = nn.Tanh
    num_fc = 1000
    num_output = 10

    def __init__(self, mode: str = "quantized") -> None:
        """builds the model, depending on mode in either quantized or full_precision mode

        Args:
            mode (str, optional): build mode for lenet model, either 'quantized' or 'full_precision'.
                Defaults to "quantized".

        Raises:
            ValueError: Thrown if unsupported mode was passed.
        """
        super(LeNet, self).__init__()
        if mode == "quantized":
            self.model = nn.Sequential(
                nn.Conv2d(1, self.num_channels_conv, kernel_size=5),
                self.activation_function(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(self.num_channels_conv),

                QActivation(),
                QConv2d(
                    self.num_channels_conv,
                    self.num_channels_conv,
                    kernel_size=5),

                nn.BatchNorm2d(self.num_channels_conv),
                nn.MaxPool2d(2, 2),

                nn.Flatten(),

                QActivation(activation="sign"),
                QLinear(self.num_channels_conv * 4 * 4, self.num_fc, quantization="sign"),
                nn.BatchNorm1d(self.num_fc),

                QLinear(self.num_fc, self.num_output, quantization="sign"),
            )
        elif mode == "full_precision":
            self.model = nn.Sequential(
                nn.Conv2d(1, self.num_channels_conv, kernel_size=5),
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
        else:
            raise ValueError(
                f"mode {mode} not supported for lenet, please choose from either 'quantized' or 'full_precision'")

    def forward(self, x: Tensor) -> Tensor:
        """Forwards the input tensor through lenet

        Args:
            x (Tensor): input tensor

        Returns:
            Tensor: forwarded tensor
        """
        return self.model(x)
