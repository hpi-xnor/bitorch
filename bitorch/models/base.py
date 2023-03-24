import logging
from argparse import ArgumentParser
from typing import Optional, List, Any, Tuple

import torch
from torch import nn
from torch.nn import Sequential

from bitorch import RuntimeMode
from bitorch.layers import convert
from bitorch.layers.qconv1d import QConv1dBase, QConv1d_NoAct
from bitorch.layers.qconv2d import QConv2dBase, QConv2d_NoAct
from bitorch.layers.qconv3d import QConv3dBase, QConv3d_NoAct
from bitorch.models.model_hub import (
    load_from_hub,
    pop_first_layers,
    pop_last_layers,
    get_input_size,
    get_output_size,
    set_layer_in_model,
)


class Model(nn.Module):
    """Base class for Bitorch models"""

    name = ""
    version_table_url = "https://api.wandb.ai/artifactsV2/default/hpi-deep-learning/QXJ0aWZhY3Q6MzE1MzQ1ODM1/a9bd2573417efc7fb8f562f06f3d322d"

    def __init__(self, input_shape: List[int], num_classes: int = 0) -> None:
        super(Model, self).__init__()
        self._model = nn.Module()
        self._input_shape = input_shape
        self._num_classes = num_classes

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        """allows additions to the argument parser if required, e.g. to add layer count, etc.

        ! please note that the inferred variable names of additional cli arguments are passed as
        keyword arguments to the constructor of this class !

        Args:
            parser (ArgumentParser): the argument parser
        """
        pass

    def model(self) -> nn.Module:
        """getter method for model

        Returns:
            Module: the main torch.nn.Module of this model
        """
        return self._model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forwards the input tensor through the model.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: the model output
        """
        return self._model(x)

    def initialize(self) -> None:
        """initializes model weights a little differently for BNNs."""
        for module in self._model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # binary layers
                if isinstance(
                    module,
                    (
                        QConv1dBase,
                        QConv2dBase,
                        QConv3dBase,
                        QConv1d_NoAct,
                        QConv2d_NoAct,
                        QConv3d_NoAct,
                    ),
                ):
                    nn.init.xavier_normal_(module.weight)
                else:
                    if module.kernel_size[0] == 7:
                        # first conv layer
                        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                    else:
                        # other 32-bit conv layers
                        nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def convert(self, new_mode: RuntimeMode, device: Optional[torch.device] = None, verbose: bool = False) -> "Model":
        return convert(self, new_mode, device, verbose)

    @classmethod
    def from_pretrained(
        cls, source: Optional[str] = None, mode: RuntimeMode = RuntimeMode.DEFAULT, **kwargs: Any
    ) -> "Model":
        """Loads a pretrained model from a file or from the model hub.

        Args:
            source (Optional[str], optional): source of the model. If omitted, the model will be loaded from model hub. Defaults to None.
            mode (RuntimeMode, optional): Runtime mode to load the model in. Defaults to RuntimeMode.DEFAULT.

        Returns:
            nn.Module: the pretrained model
        """
        model = cls(**kwargs)  # type: ignore
        if source is not None:
            logging.info(f"Loading {cls.name} model state_dict from file {source}")
            state_dict = torch.load(source)
        else:
            kwargs["model_name"] = cls.name.lower()
            logging.info(f"Downloading {cls.name} model state_dict from hub...")
            state_dict = load_from_hub(cls.version_table_url, **kwargs)

        model.load_state_dict(state_dict)
        return model

    @classmethod
    def _load_default_model(cls) -> "Model":
        raise NotImplementedError()

    @classmethod
    def as_backbone(
        cls,
        input_size: Optional[Tuple[int]] = None,
        output_size: Optional[Tuple[int]] = None,
        prepend_layers: Optional[nn.Module] = None,
        append_layers: Optional[nn.Module] = None,
        sanity_check: bool = True,
        as_feature_extractor: bool = False,
    ) -> "Model":
        """
        Creates a model that can be used as a backbone for other models.
        This method takes care of making the input and output sizes compatible with the model.
        You can change the input and output size of the model as well als add layers in front of and after the model.
        If the layers in front of or after the model are not compatible with the model, this method will
        add the necessary layers to make the model compatible.

        If you just want the model as feature extractor, only the last layer will be removed.

        After the modification a sanity check will be performed by default. This will create a random input tensor and forward it through the model.
        This check can be turned off using the sanity_check parameter.

        Args:
            input_size (Tuple[int], optional): Shape of the input (without batch dimension). Defaults to None.
            output_size (Tuple[int], optional): shape of the wanted output (without batch dimension). Defaults to None.
            prepend_layers (Sequential, optional): layers to be added in front of the model. Defaults to None.
            append_layers (Sequential, optional): layers to be added after the model. Defaults to None.
            sanity_check (bool, optional): toggles testing of created model with an example input. Defaults to True.
            as_feature_extractor (bool, optional): toggles, if the last layer of the model should be removed. Defaults to False.

        Raises:
            ValueError: thrown if both input_size and prepend_layers are specified
            ValueError: thrown if both output_size and append_layers are specified
            NotImplementedError: thrown if the connecting layers are something other than nn.Linear or nn.Conv2d
            RuntimeError: thrown if the model is not compatible with the specified input_size

        Returns:
            Model: the created model
        """
        if input_size is not None and prepend_layers is not None:
            raise ValueError("Cannot specify both input_size and prepend_layers")
        if output_size is not None and append_layers is not None:
            raise ValueError("Cannot specify both output_size and append_layers")
        model = cls._load_default_model()
        if input_size is not None:
            first_layer_name, model_output_size, removed_layer = pop_first_layers(model._model)
            if len(input_size) == 1:
                set_layer_in_model(model._model, first_layer_name, nn.Linear(input_size[0], model_output_size))
            elif len(input_size) == 3:
                set_layer_in_model(
                    model._model,
                    first_layer_name,
                    nn.Conv2d(  # type: ignore
                        input_size[0],
                        model_output_size,
                        kernel_size=removed_layer.kernel_size,  # type: ignore
                        stride=removed_layer.stride,  # type: ignore
                        padding=removed_layer.padding,  # type: ignore
                        dilation=removed_layer.dilation,  # type: ignore
                        groups=removed_layer.groups,  # type: ignore
                        bias=removed_layer.bias is not None,  # type: ignore
                        padding_mode=removed_layer.padding_mode,  # type: ignore
                    ),
                )
            else:
                raise NotImplementedError("Only 2D and 3D inputs are supported")
        elif prepend_layers is not None:
            prepend_layers = Sequential(prepend_layers)
            first_layer_name, model_input_size, removed_layer = pop_first_layers(model._model)
            prepend_output_size, prepend_output_type = get_output_size(prepend_layers)
            if prepend_output_size != model_input_size:
                logging.info("Changing output size of prepend_layers to match model")
                if prepend_output_type is None:
                    prepend_layers.append(removed_layer)
                elif issubclass(prepend_output_type, nn.Linear):
                    prepend_layers.append(nn.Linear(prepend_output_size, model_input_size))
                elif issubclass(prepend_output_type, nn.Conv2d):
                    prepend_layers.append(
                        nn.Conv2d(  # type: ignore
                            prepend_output_size,
                            model_input_size,
                            kernel_size=removed_layer.kernel_size,  # type: ignore
                            stride=removed_layer.stride,  # type: ignore
                            padding=removed_layer.padding,  # type: ignore
                            dilation=removed_layer.dilation,  # type: ignore
                            groups=removed_layer.groups,  # type: ignore
                            bias=removed_layer.bias is not None,  # type: ignore
                            padding_mode=removed_layer.padding_mode,  # type: ignore
                        )
                    )
                else:
                    raise NotImplementedError("Only 2D and 3D inputs are supported")
            set_layer_in_model(model._model, first_layer_name, prepend_layers)

        if output_size is not None:
            last_layer_name, model_output_size, removed_layer = pop_last_layers(model._model)
            if len(output_size) == 1:
                set_layer_in_model(model._model, last_layer_name, nn.Linear(model_output_size, output_size[0]))
            elif len(output_size) == 3:
                set_layer_in_model(
                    model._model,
                    last_layer_name,
                    nn.Conv2d(  # type: ignore
                        model_output_size,
                        output_size[0],
                        kernel_size=removed_layer.kernel_size,  # type: ignore
                        stride=removed_layer.stride,  # type: ignore
                        padding=removed_layer.padding,  # type: ignore
                        dilation=removed_layer.dilation,  # type: ignore
                        groups=removed_layer.groups,  # type: ignore
                        bias=removed_layer.bias is not None,  # type: ignore
                        padding_mode=removed_layer.padding_mode,  # type: ignore
                    ),
                )
            else:
                raise NotImplementedError("Only 2D and 3D inputs are supported")
        elif append_layers is not None:
            last_layer_name, model_output_size, removed_layer = pop_last_layers(model._model)
            append_input_size, append_input_type = get_input_size(append_layers)
            if append_input_size != model_output_size:
                logging.info("changing input size of append_layers to match model")
                if append_input_type is None:
                    append_layers = Sequential(removed_layer, append_layers)
                elif issubclass(append_input_type, nn.Linear):
                    append_layers = Sequential(nn.Linear(model_output_size, append_input_size), append_layers)
                elif issubclass(append_input_type, nn.Conv2d):
                    append_layers = Sequential(
                        nn.Conv2d(  # type: ignore
                            model_input_size,
                            append_input_size,
                            kernel_size=removed_layer.kernel_size,  # type: ignore
                            stride=removed_layer.stride,  # type: ignore
                            padding=removed_layer.padding,  # type: ignore
                            dilation=removed_layer.dilation,  # type: ignore
                            groups=removed_layer.groups,  # type: ignore
                            bias=removed_layer.bias is not None,  # type: ignore
                            padding_mode=removed_layer.padding_mode,  # type: ignore
                        ),
                        append_layers,
                    )
                else:
                    raise NotImplementedError("Only Linear and Conv2d layers are supported")
            set_layer_in_model(model._model, last_layer_name, append_layers)

        elif as_feature_extractor:
            pop_last_layers(model._model)

        if sanity_check:
            logging.info("running sanity check...")
            model_input_size, _ = get_input_size(model._model)
            # todo: check if input_size is given
            rand_input = torch.rand((2, model_input_size, 224, 224))
            try:
                model(rand_input)
            except Exception as e:
                logging.error(f"Missconfiguration of backbone. Sanity check failed: {e}")
                raise e
            logging.info("sanity check successful!")

        return model

    def on_train_batch_end(self, layer: nn.Module) -> None:
        """Is used with the pytorch lighting on_train_batch_end callback

        Implement it to e.g. clip weights after optimization. Is recursively applied to every submodule.

        Args:
            layer (nn.Module): current layer
        """
        pass


class NoArgparseArgsMixin:
    """
    Mixin for Models which subclass an existing Model, but do not have any argparse arguments anymore.

    By using this Mixin, there is no special Parser displayed for the class.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @staticmethod
    def add_argparse_arguments(parser: ArgumentParser) -> None:
        pass
