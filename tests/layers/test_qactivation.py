import pytest
from bitorch.layers.qactivation import QActivation
from bitorch.layers.config import config
from bitorch.quantizations import Sign


activation = QActivation()


def test_qactivation():
    assert isinstance(activation.activation, type(config.default_quantization()))
    assert isinstance(QActivation("sign").activation, Sign)
    assert isinstance(QActivation(Sign(3.0)).activation, Sign)
    with pytest.raises(ValueError):
        QActivation("iNvAlIdNaMe")
