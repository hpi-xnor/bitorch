import math
import torch
import matplotlib


class Debug_Layer(torch.nn.Module):

    def __init__(self,
                 module: torch.nn.Module,
                 print_debug: bool = False,
                 graphic_debug: bool = False,
                 subplot: matplotlib.pyplot = None) -> None:
        """Initializes debug module, flags and objects.

        Args:
            module (torch.nn.Module): the module of which the weights shall be debugged
            print_debug (bool, optional): Toggles stdout debugging of weights. Defaults to False.
            graphic_debug (bool, optional): Toggles graphic debbuging in given subplot. Defaults to False.
            subplot (matplotlib.pyplot, optional): subplot to output graphic output to.
                only needed if graphic_debug is set to True. Defaults to None.

        Raises:
            ValueError: Raised if graphic output is enabled but no subplot was passed.
        """
        self._debug_module = module
        self._print_debug = print_debug
        self._graphic_debug = graphic_debug
        self._subplot = subplot
        if graphic_debug and subplot is None:
            raise ValueError("graphic debug is activated but no subplot was passed!")

    def forward(self, x: torch.Tensor):
        """Forwards the input tensor and outputs debug information about the given modules weights.

        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.

        Raises:
            ValueError: Raised if graphic output is enabled but no subplot was passed.

        Returns:
            torch.Tensor: the input tensor
        """
        weight = self._debug_module.weight
        if not hasattr(self._debug_module, 'quantize'):
            weight = self._debug_module.quantize(weight)
        if self._print_debug:
            print(weight)
        if self._graphic_debug:
            if self._subplot is None:
                raise ValueError("no subplot given to debug into!")
            num_filters = weight.shape[0] * weight.shape[1]
            for filter_index in range(num_filters):
                filter_index_a = math.floor(filter_index / weight.shape[0])
                filter_index_b = filter_index % weight.shape[1]
                filter_to_show = weight[filter_index_a, filter_index_b]
                # TODO: show filter in subplot
        return x
