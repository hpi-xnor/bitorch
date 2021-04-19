"""Module containting the quantized dense layer"""

import torch
from torch.nn import Linear
from torch.nn.functional import linear

from . import layerconfig


class QDense(Linear):
    def __init__(self, *kargs, quantization: str = None, **kwargs):
        super(QDense, self).__init__(*kargs, **kwargs)
        self.quantize = layerconfig.config.get_quantization_function(quantization)

    def forward(self, x: torch.Tensor):
        """Forwards x through the binary linear layer. This code is inspired from the binarizedlinear-implementation at
        https://github.com/itayhubara/BinaryNet.pytorch/blob/master/models/binarized_modules.py

        Args:
            x (torch.Tensor): tensor to forward

        Returns:
            torch.Tensors: forwarded tensor
        """

        # x = self.quantize(x)
        # if not hasattr(self.weight, 'org'):
        #     self.weight.org = self.weight.data.clone()
        # self.weight.data = self.quantize(self.weight.org)
        x = linear(x, self.quantize(self.weight), self.bias)
        # if self.bias is not None:
        #     self.bias.org = self.bias.data.clone()
        #     x += self.bias.view(1, -1).expand_as(x)
        return x
