import pytest
from torch._C import Value
from bitorch.layers.qactivation import QActivation
from bitorch.layers import layerconfig


activation = QActivation()


def test_qactivation():
    assert isinstance(activation.activation, type(layerconfig.config.default_activation()))
    with pytest.raises(ValueError):
        QActivation("iNvAlIdNaMe")
