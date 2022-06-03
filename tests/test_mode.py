import pytest

import bitorch
from bitorch import RuntimeMode


def test_mode_creation_from_name():
    for mode_str in RuntimeMode.__members__.keys():
        assert isinstance(RuntimeMode.from_string(mode_str), RuntimeMode)


def test_mode_supports_self():
    for mode in RuntimeMode.__members__.values():
        assert mode.is_supported_by(mode)


def test_mode_does_not_support_other_mode():
    for mode in RuntimeMode.__members__.values():
        for other_mode in RuntimeMode.__members__.values():
            if mode == other_mode:
                continue
            assert not mode.is_supported_by(other_mode)


def test_mode_self_addition():
    for mode in RuntimeMode.__members__.values():
        same_mode_twice = mode + mode
        assert same_mode_twice == mode


def test_mode_addition_supports_both():
    for mode in RuntimeMode.__members__.values():
        for other_mode in RuntimeMode.__members__.values():
            if mode == other_mode:
                continue
            added_modes = mode + other_mode
            assert mode.is_supported_by(added_modes)
            assert other_mode.is_supported_by(added_modes)


def test_str_output():
    assert str(RuntimeMode.DEFAULT) == "default"


def test_repr_output():
    assert repr(RuntimeMode.DEFAULT) == "<RuntimeMode.DEFAULT: 1>"


def test_bitorch_default_mode():
    assert bitorch.mode == RuntimeMode.DEFAULT


@pytest.fixture()
def set_inference_mode():
    bitorch.mode = RuntimeMode.INFERENCE_AUTO
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_set_bitorch_mode(set_inference_mode):
    assert bitorch.mode == RuntimeMode.INFERENCE_AUTO


@pytest.fixture()
def reset_modes():
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_setting_different_modes(reset_modes):
    assert bitorch.mode == RuntimeMode.DEFAULT
    bitorch.mode = RuntimeMode.INFERENCE_AUTO
    assert bitorch.mode == RuntimeMode.INFERENCE_AUTO
