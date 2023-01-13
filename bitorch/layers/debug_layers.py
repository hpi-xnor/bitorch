from typing import Optional, Any
import torch
from .config import config


class _Debug(torch.nn.Module):
    def __init__(self, debug_interval: int = 100, num_outputs: int = 10, name: str = "Debug") -> None:
        """inits values.

        Args:
            debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
            num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.
            name (str, optional): name of debug layer, only relevant for print debugging
        """
        super(_Debug, self).__init__()
        self._debug_interval = debug_interval
        self._num_outputs = num_outputs
        self.name = name

        self._forward_counter = 0

    def _debug(self, x: torch.Tensor) -> None:
        pass

    def _debug_tensor(self, debug_tensor: torch.Tensor) -> None:
        """outputs debug information for given tensor

        Args:
            debug_tensor (torch.Tensor): tensor to be debugged
        """
        if config.debug_activated and self._forward_counter % self._debug_interval == 0:
            self._debug(debug_tensor)

        self._forward_counter += 1


class _PrintDebug(_Debug):
    def _debug(self, debug_tensor: torch.Tensor) -> None:
        """prints the first num_outputs entries in tensor debug_tensor

        Args:
            debug_tensor (torch.Tensor): tensor to be debugged
        """
        print(
            self.name, ":", debug_tensor if len(debug_tensor) < self._num_outputs else debug_tensor[: self._num_outputs]
        )


class _GraphicalDebug(_Debug):
    def __init__(
        self,
        figure: Optional[object] = None,
        images: Optional[list] = None,
        debug_interval: int = 100,
        num_outputs: int = 10,
    ) -> None:
        """Debugs the given layer by drawing weights/inputs in given matplotlib plot images.

        Args:
            figure (object): figure to draw in
            images (list): list of images to update with given data
            debug_interval (int, optional): interval at which debug output shall be prompted. Defaults to 100.
            num_outputs (int, optional): number of weights/inputs that shall be debugged. Defaults to 10.

        Raises:
            ValueError: raised if number of images does not match desired number of outputs.
        """
        super(_GraphicalDebug, self).__init__(debug_interval, num_outputs)
        self.set_figure(figure)
        self.set_images(images)

    def set_figure(self, figure: Optional[object] = None) -> None:
        """setter for figure object

        Args:
            figure (object): the figure object
        """
        self._figure = figure

    def set_images(self, images: Optional[list] = None) -> None:
        """setter for images list

        Args:
            images (list): list of image objects to output the graphical information to

        Raises:
            ValueError: raised if number of images does not match desired number of outputs.
        """
        self._images = images
        if self._images is not None and len(self._images) != self._num_outputs:
            raise ValueError(
                f"number of given images ({len(self._images)}) must match "
                f"number of desired outputs ({self._num_outputs})!"
            )

    def _debug(self, debug_tensor: torch.Tensor) -> None:
        """draws graphical debug information about given debug tensor into figure

        Args:
            debug_tensor (torch.Tensor): tensor to be debugged

        Raises:
            ValueError: raised if either no figure or no images were given
        """
        debug_tensor = debug_tensor.clone().detach()
        if self._figure is None or self._images is None:
            raise ValueError("no subplot given to debug into!")
        dimensionality = len(debug_tensor.shape)
        filters = []
        # depending of dimensionality select the filters to be drawn to the images
        if dimensionality == 2:
            filters.append(debug_tensor)
        elif dimensionality == 3 or dimensionality == 4:
            for i in range(debug_tensor.shape[0]):
                for j in range(debug_tensor.shape[1]):
                    filters.append(debug_tensor[i, j])
        elif dimensionality == 5:
            for i in range(debug_tensor.shape[0]):
                for j in range(debug_tensor.shape[1]):
                    for k in range(debug_tensor.shape[2]):
                        filters.append(debug_tensor[i, j, k])
        # normalize all filters
        for image, fltr in zip(self._images, filters):
            fltr -= torch.min(fltr)
            fltr /= torch.max(fltr)
            image.set_data(fltr)
        self._figure.canvas.draw()


"""
Classes above are internal, use classes below for debugging
"""


class InputPrintDebug(_PrintDebug):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the given tensor without modification, debug output if activated

        Args:
            x (torch.Tensor): tensor to be debugged

        Returns:
            torch.Tensor: input tensor x
        """
        self._debug_tensor(x)
        return x


class InputGraphicalDebug(_GraphicalDebug):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the given tensor without modification, debug output if activated

        Args:
            x (torch.Tensor): tensor to be debugged

        Returns:
            torch.Tensor: input tensor x
        """
        self._debug_tensor(x)
        return x


class WeightPrintDebug(_PrintDebug):
    def __init__(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        """stores given module

        Args:
            module (torch.nn.Module): module the weights of which shall be debugged
        """
        super(WeightPrintDebug, self).__init__(*args, **kwargs)
        self._debug_module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input tensor through the debug model and outputs debug information about the given modules weights.

        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.

        Returns:
            torch.Tensor: the input tensor
        """
        x = self._debug_module(x)

        weight = self._debug_module.weight.clone()  # type: ignore
        # check if given module is a quantized module
        if hasattr(self._debug_module, "quantize"):
            weight = self._debug_module.quantize(weight)  # type: ignore
        self._debug_tensor(weight)

        return x


class WeightGraphicalDebug(_GraphicalDebug):
    def __init__(self, module: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        """stores given module

        Args:
            module (torch.nn.Module): module the weights of which shall be debugged
        """
        super(WeightGraphicalDebug, self).__init__(*args, **kwargs)
        self._debug_module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forwards the input tensor through the debug model and outputs debug information about the given modules weights.

        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.

        Returns:
            torch.Tensor: the input tensor
        """
        x = self._debug_module(x)

        weight = self._debug_module.weight.clone()  # type: ignore
        # check if given module is a quantized module
        if hasattr(self._debug_module, "quantize"):
            weight = self._debug_module.quantize(weight)  # type: ignore
        self._debug_tensor(weight)

        return x


class ShapePrintDebug(_PrintDebug):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """prints the shape of x, leaves x untouched

        Args:
            x (torch.Tensor): the tensor to be debugged

        Returns:
            torch.Tensor: input tensor x
        """
        self._debug_tensor(torch.tensor(x.shape))
        return x
