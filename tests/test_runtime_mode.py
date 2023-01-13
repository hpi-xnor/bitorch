import pytest

import bitorch
from bitorch import RuntimeMode


TEST_MODE = RuntimeMode.INFERENCE_AUTO


def test_mode_creation_from_name():
    for mode_str in RuntimeMode.list_of_names():
        assert isinstance(RuntimeMode.from_string(mode_str), RuntimeMode)


def test_mode_supports_self():
    for mode in RuntimeMode.available_values():
        assert mode.is_supported_by(mode)


def test_mode_does_not_support_other_mode():
    for mode in RuntimeMode.available_values():
        for other_mode in RuntimeMode.available_values():
            if mode == other_mode or mode == RuntimeMode.RAW or other_mode == RuntimeMode.RAW:
                continue
            assert not mode.is_supported_by(other_mode)


def test_mode_self_addition():
    for mode in RuntimeMode.available_values():
        same_mode_twice = mode + mode
        assert same_mode_twice == mode


def test_mode_addition_supports_both():
    for mode in RuntimeMode.available_values():
        for other_mode in RuntimeMode.available_values():
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
def set_test_mode():
    bitorch.mode = TEST_MODE
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_set_bitorch_mode(set_test_mode):
    assert bitorch.mode == TEST_MODE


@pytest.fixture()
def reset_modes():
    bitorch.mode = RuntimeMode.DEFAULT
    yield None
    bitorch.mode = RuntimeMode.DEFAULT


def test_setting_different_modes(reset_modes):
    assert bitorch.mode == RuntimeMode.DEFAULT
    bitorch.mode = TEST_MODE
    assert bitorch.mode == TEST_MODE


def test_with_statement(reset_modes):
    with bitorch.change_mode(TEST_MODE):
        assert bitorch.mode == TEST_MODE


def test_pause_wrap(reset_modes):
    with bitorch.pause_wrapping():
        assert bitorch.mode == RuntimeMode.RAW
