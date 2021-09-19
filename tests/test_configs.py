from bitorch import configs_by_name
import pytest

TEST_INPUT_DATA = list(configs_by_name.items())


@pytest.mark.parametrize("config_name, config_obj", TEST_INPUT_DATA)
def test_config(config_name, config_obj):
    assert config_name == config_obj.name

    for attribute in config_obj._configurable_attributes:
        attribute_value = getattr(config_obj, attribute)
        assert f"get_{attribute}" in dir(config_obj)
        assert f"set_{attribute}" in dir(config_obj)

        assert getattr(config_obj, f"get_{attribute}")(config_obj) == attribute_value

        getattr(config_obj, f"set_{attribute}")(config_obj, 42)
        assert getattr(config_obj, attribute) == 42

        getattr(config_obj, f"set_{attribute}")(config_obj, attribute_value)
        assert getattr(config_obj, attribute) == attribute_value
        assert getattr(config_obj, f"get_{attribute}")(config_obj) == attribute_value
