import pytest
from bitorch.layers.qactivation import QActivation
from bitorch.layers.config import config
from bitorch.quantizations import Sign


activation = QActivation()


def test_qactivation():
    assert isinstance(activation._activation, type(config.input_quantization()))
    assert isinstance(QActivation("sign")._activation, Sign)
    assert isinstance(QActivation(Sign())._activation, Sign)
    with pytest.raises(ValueError):
        QActivation("iNvAlIdNaMe")
