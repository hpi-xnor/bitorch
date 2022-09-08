import sched
from torch.nn import Module
import torch
from typing import List, Type
from .base import Quantization
from copy import deepcopy
from .config import config


class MixLinearScheduling(Quantization):
    name = "__mixlinarscheduling__"
    bit_width = 32

    def __init__(self, quantizations, steps):
        super().__init__()
        self.quantizations = [deepcopy(quantization) for quantization in quantizations]
        self.bit_width = self.quantizations[-1].bit_width
        self.mix_factor = 0.0
        self.step_count = 0
        self.steps = steps

    def step(self):
        self.step_count += 1
        self.mix_factor = self.step_count / self.steps
        self.mix_factor = min(self.mix_factor, 1.0)

    def quantize(self, x: torch.Tensor):
        if len(self.quantizations) == 1:
            return self.quantizations[0](x)
        scaled_mix_factor = self.mix_factor * (len(self.quantizations) - 1)
        lower_idx = int(scaled_mix_factor)
        higher_idx = lower_idx + 1
        if higher_idx == len(self.quantizations):
            return self.quantizations[lower_idx](x)

        inter_unit_mix_factor = scaled_mix_factor - lower_idx
        return self.quantizations[higher_idx](x) * inter_unit_mix_factor + self.quantizations[lower_idx](x) * (1.0 - inter_unit_mix_factor)


class Quantization_Scheduler(Module):

    procedure_classes = {
        "mix_linear": MixLinearScheduling,
    }

    def __init__(self, model, steps, quantizations: List[Quantization], scheduling_procedure: str, exclude_layers: List[Type] = []):
        super().__init__()

        assert steps > 0, "steps has to be an integer > 0"
        assert isinstance(quantizations, list)
        assert len(quantizations) > 0

        self.quantizations = quantizations
        self.steps = steps

        self.scheduled_quantizer = self.get_scheduled_quantizer(scheduling_procedure)

        self.scheduled_quantizer_instances = []
        self.replace_quantizations(model, exclude_layers)

    def get_scheduled_quantizer(self, procedure):
        return self.procedure_classes[procedure]

    def replace_quantizations(self, model, exclude_layers):
        for name in dir(model):
            module = getattr(model, name)
            if issubclass(type(module), Quantization):
                self.scheduled_quantizer_instances.append(self.scheduled_quantizer(self.quantizations, self.steps))
                setattr(model, name, self.scheduled_quantizer_instances[-1])

        for child in model.children():
            if type(child) not in exclude_layers:
                self.replace_quantizations(child, exclude_layers)

    def step(self):
        for scheduled_quantizer in self.scheduled_quantizer_instances:
            scheduled_quantizer.step()
