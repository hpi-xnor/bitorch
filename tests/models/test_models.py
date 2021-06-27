from bitorch.datasets import CIFAR10, dataset_from_name
from bitorch.models import models_by_name, LeNet, Resnet
import torch
import numpy as np
import pytest

TEST_INPUT_DATA = [
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 18}, (100, 1, 28, 28)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 18}, (100, 3, 32, 32)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 18}, (2, 3, 244, 244)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 34}, (100, 1, 28, 28)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 34}, (100, 3, 32, 32)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 34}, (2, 3, 244, 244)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 50}, (10, 1, 28, 28)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 50}, (10, 3, 32, 32)),
    (Resnet, {"resnet_version": 1, "resnet_num_layers": 50}, (1, 3, 244, 244)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 18}, (100, 1, 28, 28)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 18}, (100, 3, 32, 32)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 18}, (2, 3, 244, 244)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 34}, (100, 1, 28, 28)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 34}, (100, 3, 32, 32)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 34}, (2, 3, 244, 244)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 50}, (10, 1, 28, 28)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 50}, (10, 3, 32, 32)),
    (Resnet, {"resnet_version": 2, "resnet_num_layers": 50}, (1, 3, 244, 244)),
    (LeNet, {"bits": 1}, (100, 1, 28, 28)),
    (LeNet, {"bits": 32}, (100, 1, 28, 28)),
]


@pytest.mark.parametrize("model_class, kwargs, input_shape", TEST_INPUT_DATA)
def test_models(model_class, kwargs, input_shape) -> None:
    dataset = dataset_from_name("cifar10")
    assert dataset is CIFAR10
    dataset.shape = input_shape

    assert models_by_name[model_class.name] is model_class

    model = model_class(dataset=dataset, **kwargs)
    input_values = torch.Tensor(np.random.uniform(0, 1.0, input_shape))
    model(input_values)
