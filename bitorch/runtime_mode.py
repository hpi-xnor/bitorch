from enum import Enum
from typing import Union, Any
from functools import total_ordering


runtime_mode_type = Union["RuntimeMode", int]


@total_ordering
class RuntimeMode(Enum):
    DEFAULT = 1
    TRAIN = 2
    CPP_INFERENCE = 4
    GPU_INFERENCE = 8

    def __add__(self, other: runtime_mode_type) -> runtime_mode_type:
        if self._to_int(self) == self._to_int(other):
            return self
        return self._to_int(other) + self.value

    @staticmethod
    def _max_val() -> int:
        return sum(map(lambda x: x.value, RuntimeMode.__members__.values()))

    @staticmethod
    def is_single_mode(mode: runtime_mode_type) -> bool:
        return any(x.value == mode for x in RuntimeMode.__members__.values())

    @staticmethod
    def is_combined_mode(mode: runtime_mode_type) -> bool:
        return 0 < mode < RuntimeMode._max_val()

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, RuntimeMode) and not isinstance(other, int):
            return NotImplemented
        return self.value < self._to_int(other)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RuntimeMode) and not isinstance(other, int):
            return NotImplemented
        return self.value == self._to_int(other)

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def _to_int(mode: runtime_mode_type) -> int:
        if isinstance(mode, RuntimeMode):
            return mode.value
        return mode

    @staticmethod
    def from_string(level: str) -> "RuntimeMode":
        return {
            "default": RuntimeMode.DEFAULT,
            "train": RuntimeMode.TRAIN,
            "cpp_inference": RuntimeMode.CPP_INFERENCE,
            "gpu_inference": RuntimeMode.GPU_INFERENCE,
        }[level.lower()]

    @staticmethod
    def mode_compatible(required_mode: "RuntimeMode", provided_modes: runtime_mode_type) -> bool:
        return bool(RuntimeMode._to_int(required_mode) & RuntimeMode._to_int(provided_modes))

    def is_supported_by(self, provided_modes: runtime_mode_type) -> bool:
        return self.mode_compatible(self, provided_modes)
