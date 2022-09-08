from torch.nn import Module
import torch
from typing import List, Type
from .base import Quantization
from copy import deepcopy


class ScheduledQuantizer(Quantization):
    """Base class for scheduled quantizers to inherit from"""

    name = "__scheduledquantizer__"
    bit_width = 32

    def __init__(self, quantizations: List[Quantization], steps: int) -> None:
        """Initias scheduled optimizer and sets bitwidth to width of last quantization to be scheduled.

        Args:
            quantizations (List[Quantization]): list of quantizations to be scheduled
            steps (int): number of steps. at the end of each step, the step() method has to be called once.
        """
        super().__init__()
        self.quantizations = [deepcopy(quantization) for quantization in quantizations]
        self.bit_width = self.quantizations[-1].bit_width
        self.step_count = 0
        self.factor = 0.0
        self.steps = steps

    def step(self) -> None:
        """increments step count and updates internal factor variable"""
        self.step_count += 1
        self.factor = self.step_count / self.steps
        self.factor = min(self.factor, 1.0)


class MixLinearScheduling(ScheduledQuantizer):
    name = "__mixlinarscheduling__"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """interpolates linearly between the output of the specified quantizations.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized output tensor
        """
        if len(self.quantizations) == 1:
            return self.quantizations[0](x)

        scaled_mix_factor = self.factor * (len(self.quantizations) - 1)
        lower_idx = int(scaled_mix_factor)
        higher_idx = lower_idx + 1
        if higher_idx == len(self.quantizations):
            return self.quantizations[lower_idx](x)

        inter_unit_mix_factor = scaled_mix_factor - lower_idx
        return self.quantizations[higher_idx](x) * inter_unit_mix_factor + self.quantizations[lower_idx](x) * (
            1.0 - inter_unit_mix_factor
        )


class StepScheduling(ScheduledQuantizer):
    name = "__stepscheduling__"

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """interpolates linearly between the output of the specified quantizations.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: quantized output tensor
        """
        quantization_idx = min(int(self.factor * len(self.quantizations)), len(self.quantizations) - 1)
        return self.quantizations[quantization_idx](x)


class Quantization_Scheduler(Module):

    procedure_classes = {"mix_linear": MixLinearScheduling, "step": StepScheduling}

    def __init__(
        self,
        model: Module,
        steps: int,
        quantizations: List[Quantization],
        scheduling_procedure: str,
        exclude_layers: List[Type] = [],
    ) -> None:
        """Initiates the quantization scheduler and replaces the activation function inside the model with scheduled
        quantizers

        Args:
            model (Module): model to be scheduled quantized
            steps (int): number of steps, e.g. number of epochs. Each step the step() method has to be called once to
                update all scheduled quantizers.
            quantizations (List[Quantization]): Quantization functions to be scheduled
            scheduling_procedure (str): procedure to be used for scheduling. See available subclasses of
                ScheduledQuantizer
            exclude_layers (List[Type], optional): list of layers types to exclude from replacement with scheduled
                quantizers. Defaults to [].
        """
        super().__init__()

        assert steps > 0, "steps has to be an integer > 0"
        assert isinstance(quantizations, list)
        assert len(quantizations) > 0

        self.quantizations = quantizations
        self.steps = steps

        self.scheduled_quantizer = self.get_scheduled_quantizer(scheduling_procedure)

        self.scheduled_quantizer_instances: List[ScheduledQuantizer] = []
        self.replace_quantizations(model, exclude_layers)

    def get_scheduled_quantizer(self, procedure: str) -> Type:
        return self.procedure_classes[procedure]

    def replace_quantizations(self, model: Module, exclude_layers: List[Type]) -> None:
        """replaces all quantization functions present in the model with a scheduled quantizer.
        iterates recursevely to the model layers.

        Args:
            model (Module): model have the quantization functions replaced
            exclude_layers (List[Type]): list of layers to exclude from replacement, e.g. if QConv2d is specified,
                the quantization functions from all QConv2d layers (input and weight) are not replaced
        """
        for name in dir(model):
            module = getattr(model, name)
            if issubclass(type(module), Quantization):
                self.scheduled_quantizer_instances.append(self.scheduled_quantizer(self.quantizations, self.steps))
                setattr(model, name, self.scheduled_quantizer_instances[-1])

        for child in model.children():
            if type(child) not in exclude_layers:
                self.replace_quantizations(child, exclude_layers)

    def step(self) -> None:
        """updates all instances of scheduled quantizers in the model"""
        for scheduled_quantizer in self.scheduled_quantizer_instances:
            scheduled_quantizer.step()
