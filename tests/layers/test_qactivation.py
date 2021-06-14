import pytest
from bitorch.layers.qactivation import QActivation
from bitorch.layers import layerconfig


activation = QActivation()


def test_qactivation():
    assert isinstance(activation.activation, type(layerconfig.config.default_quantization()))
    with pytest.raises(ValueError):
        QActivation("iNvAlIdNaMe")
