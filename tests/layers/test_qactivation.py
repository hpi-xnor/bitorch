import pytest
import torch
from bitorch.layers.qactivation import QActivation
from bitorch.layers.config import config
from bitorch.quantizations import Sign


activation = QActivation()

TEST_DATA = [-3.0, -1.0, -0.3, 0.0, 0.3, 1.0, 3.0]
TEST_THRESHOLDS = [0.0, 0.5, 1.0, 2.0]


@pytest.mark.parametrize("threshold", TEST_THRESHOLDS)
def test_qactivation(threshold):
    assert isinstance(activation._activation, type(config.input_quantization()))
    assert isinstance(QActivation("sign")._activation, Sign)
    assert isinstance(QActivation(Sign())._activation, Sign)
    with pytest.raises(ValueError):
        QActivation("iNvAlIdNaMe")

    x = torch.Tensor(TEST_DATA).float().requires_grad_(True)
    activation._gradient_cancellation_threshold = threshold

    y = activation(x)
    y.backward(x)

    if threshold > 0:
        expected_gradient = torch.where(torch.abs(x) <= threshold, x, torch.tensor(0.))
    else:
        expected_gradient = x.clone()
    assert torch.equal(expected_gradient, x.grad)
