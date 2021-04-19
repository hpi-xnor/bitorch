"""Module containing the quantized convolution layer"""

from torch.nn import Conv1d


class QConvolution(Conv1d):
    def __init__(self) -> None:
        super.__init__(QConvolution, self)
