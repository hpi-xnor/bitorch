from enum import Enum
from functools import total_ordering
from types import TracebackType
from typing import Union, Any, Optional, Type, List

import bitorch

__all__ = ["RuntimeMode", "runtime_mode_type", "change_mode", "pause_wrapping"]

runtime_mode_type = Union["RuntimeMode", int]


@total_ordering
class RuntimeMode(Enum):
    """
    Enum for BITorch modes:

    - DEFAULT: use the default implementation of all layers
    - CPU: use layer implementations for inference on CPU
    - GPU: use layer implementations for inference on GPU
    - INFERENCE_AUTO: use an automatic layer that uses the fastest implementation available (not recommended)
    - RAW: while in this mode, new layers are created as the default implementation BUT without wrapping, so they can
      not be switched to other layers later on (it does not influence already wrapped layers)
    """

    RAW = 0
    DEFAULT = 1
    CPU = 2
    GPU = 4
    INFERENCE_AUTO = 8

    def __add__(self, other: runtime_mode_type) -> runtime_mode_type:
        if self._to_int(self) == self._to_int(other):
            return self
        return self._to_int(other) + self.value

    @staticmethod
    def available_values() -> List["RuntimeMode"]:
        return RuntimeMode.__members__.values()  # type:ignore

    @staticmethod
    def list_of_names() -> List[str]:
        return RuntimeMode.__members__.keys()  # type:ignore

    @staticmethod
    def _max_val() -> int:
        return sum(map(lambda x: x.value, RuntimeMode.__members__.values()))

    @staticmethod
    def is_single_mode(mode: runtime_mode_type) -> bool:
        return any(x.value == mode for x in RuntimeMode.__members__.values())

    @staticmethod
    def is_combined_mode(mode: runtime_mode_type) -> bool:
        return 0 <= mode < RuntimeMode._max_val()

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
            "raw": RuntimeMode.RAW,
            "default": RuntimeMode.DEFAULT,
            "cpu": RuntimeMode.CPU,
            "gpu": RuntimeMode.GPU,
            "inference_auto": RuntimeMode.INFERENCE_AUTO,
        }[level.lower()]

    @staticmethod
    def mode_compatible(required_mode: "RuntimeMode", provided_modes: runtime_mode_type) -> bool:
        if required_mode == RuntimeMode.RAW.value or provided_modes == RuntimeMode.RAW.value:
            return True
        return bool(RuntimeMode._to_int(required_mode) & RuntimeMode._to_int(provided_modes))

    def is_supported_by(self, provided_modes: runtime_mode_type) -> bool:
        if self._to_int(self) == RuntimeMode.RAW.value:
            return True
        return self.mode_compatible(self, provided_modes)


class _PauseWrapping:
    def __init__(self) -> None:
        self._previous_mode = bitorch.mode

    def __enter__(self) -> "_PauseWrapping":
        bitorch.mode = RuntimeMode.RAW
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        bitorch.mode = self._previous_mode


class _SafeModeChanger:
    def __init__(self, new_mode: RuntimeMode) -> None:
        assert new_mode.is_single_mode(new_mode)
        self._previous_mode = bitorch.mode
        self._new_mode = new_mode

    def __enter__(self) -> "_SafeModeChanger":
        bitorch.mode = self._new_mode
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        bitorch.mode = self._previous_mode


change_mode = _SafeModeChanger

pause_wrapping = _PauseWrapping
