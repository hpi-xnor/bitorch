import torch
import matplotlib


class Debug_Weight_Layer(torch.nn.Module):

    def __init__(self,
                 module: torch.nn.Module,
                 print_debug: bool = False,
                 graphic_debug: bool = False,
                 figure: matplotlib.figure.Figure = None,
                 images: list = None,
                 debug_interval: int = 100) -> None:
        """Initializes debug module, flags and objects.

        Args:
            module (torch.nn.Module): the module to be wrapped in this class and of which the weights shall be debugged
            print_debug (bool, optional): Toggles stdout debugging of weights. Defaults to False.
            graphic_debug (bool, optional): Toggles graphic debbuging in given subplot. Defaults to False.
            figure (matplotlib.figure.Figure, optional): the figure to redraw after updating the images.
                Only needed if graphic debug is set to True. Defaults to None.
            images (List[plt], optional): list of images to output the filters into.
                only needed if graphic_debug is set to True. Defaults to None.
            debug_interval (int, optional): interval at which debug shall be outputed. Defaults to 100

        Raises:
            ValueError: Raised if graphic output is enabled but no images were passed.
        """
        super(Debug_Weight_Layer, self).__init__()
        self._debug_module = module
        self._print_debug = print_debug
        self._graphic_debug = graphic_debug
        self._figure = figure
        self._images = images
        self._debug_interval = debug_interval
        if graphic_debug and (images is None or figure is None):
            raise ValueError("graphic debug is activated but no images or figure were passed!")
        if self._images is not None:
            self._num_filters = len(self._images)

        self._forward_counter = 0

    def forward(self, x: torch.Tensor):
        """Forwards the input tensor through the debug model and outputs debug information about the given modules weights.

        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.

        Raises:
            ValueError: Raised if graphic output is enabled but no subplot was passed.

        Returns:
            torch.Tensor: the input tensor
        """
        x = self._debug_module(x)

        if self._forward_counter % self._debug_interval == 0:
            weight = self._debug_module.weight.clone()

            # check if given module is a quantized module
            if hasattr(self._debug_module, 'quantize'):
                weight = self._debug_module.quantize(weight)
            weight.detach_()
            if self._print_debug:
                print(weight)
            if self._graphic_debug:
                if self._figure is None or self._images is None:
                    raise ValueError("no subplot given to debug into!")
                dimensionality = len(weight.shape)
                filters = []
                if dimensionality == 2:
                    filters.append(weight)
                elif dimensionality == 3 or dimensionality == 4:
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):
                            filters.append(weight[i, j])
                elif dimensionality == 5:
                    for i in range(weight.shape[0]):
                        for j in range(weight.shape[1]):
                            for k in range(weight.shape[2]):
                                filters.append(weight[i, j, k])
                for image, fltr in zip(self._images, filters):
                    fltr -= torch.min(fltr)
                    fltr /= torch.max(fltr)
                    image.set_data(fltr)
                if isinstance(self._figure, list):
                    self._figure[0].canvas.draw()
                else:
                    self._figure.canvas.draw()

        self._forward_counter += 1
        return x


class Debug_Input_Layer(torch.nn.Module):

    def __init__(self,
                 print_debug: bool = False,
                 graphic_debug: bool = False,
                 figure: matplotlib.figure.Figure = None,
                 images: list = None,
                 debug_interval: int = 100) -> None:
        """Initializes debug module, flags and objects.

        Args:
            module (torch.nn.Module): the module to be wrapped in this class and of which the weights shall be debugged
            print_debug (bool, optional): Toggles stdout debugging of weights. Defaults to False.
            graphic_debug (bool, optional): Toggles graphic debbuging in given subplot. Defaults to False.
            figure (matplotlib.figure.Figure, optional): the figure to redraw after updating the images.
                Only needed if graphic debug is set to True. Defaults to None.
            images (List[plt], optional): list of images to output the filters into.
                only needed if graphic_debug is set to True. Defaults to None.
            debug_interval (int, optional): interval at which debug shall be outputed. Defaults to 100

        Raises:
            ValueError: Raised if graphic output is enabled but no images were passed.
        """
        super(Debug_Input_Layer, self).__init__()
        self._print_debug = print_debug
        self._graphic_debug = graphic_debug
        self._figure = figure
        self._images = images
        self._debug_interval = debug_interval
        if graphic_debug and (images is None or figure is None):
            raise ValueError("graphic debug is activated but no images or figure were passed!")
        if self._images is not None:
            self._num_features = len(self._images)

        self._forward_counter = 0

    def forward(self, x: torch.Tensor):
        """Forwards the input tensor through the debug model and outputs debug information about the given modules weights.

        Args:
            x (torch.Tensor): the Tensor to be forwarded untouched.

        Raises:
            ValueError: Raised if graphic output is enabled but no subplot was passed.

        Returns:
            torch.Tensor: the input tensor
        """

        if self._forward_counter % self._debug_interval == 0:
            x_values = x.clone()
            x_values.detach_()
            if self._print_debug:
                print(x_values)
            if self._graphic_debug:
                if self._figure is None or self._images is None:
                    raise ValueError("no subplot given to debug into!")
                dimensionality = len(x.shape)
                filters = []
                if dimensionality == 2:
                    filters.append(x_values[1])
                elif dimensionality == 3 or dimensionality == 4:
                    for i in range(x_values.shape[0]):
                        if len(filters) >= self._num_features:
                            break
                        for j in range(x_values.shape[1]):
                            if len(filters) >= self._num_features:
                                break
                            filters.append(x_values[i, j])
                elif dimensionality == 5:
                    for i in range(x_values.shape[0]):
                        if len(filters) >= self._num_features:
                            break
                        for j in range(x_values.shape[1]):
                            if len(filters) >= self._num_features:
                                break
                            for k in range(x_values.shape[2]):
                                if len(filters) >= self._num_features:
                                    break
                                filters.append(x_values[i, j, k])
                for image, fltr in zip(self._images, filters):
                    fltr -= torch.min(fltr)
                    fltr /= torch.max(fltr)
                    image.set_data(fltr)
                if isinstance(self._figure, list):
                    self._figure[0].canvas.draw()
                else:
                    self._figure.canvas.draw()

        self._forward_counter += 1
        return x
