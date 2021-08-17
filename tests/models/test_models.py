from bitorch.datasets import CIFAR10, dataset_from_name
from bitorch.models import (
    models_by_name,
    LeNet,
    Resnet,
    Resnet18_v1,
    Resnet34_v1,
    Resnet50_v1,
    Resnet18_v2,
    Resnet34_v2,
    Resnet50_v2
)
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

    (Resnet18_v1, {}, (100, 1, 28, 28)),
    (Resnet18_v1, {}, (100, 3, 32, 32)),
    (Resnet18_v1, {}, (2, 3, 244, 244)),
    (Resnet34_v1, {}, (100, 1, 28, 28)),
    (Resnet34_v1, {}, (100, 3, 32, 32)),
    (Resnet34_v1, {}, (2, 3, 244, 244)),
    (Resnet50_v1, {}, (10, 1, 28, 28)),
    (Resnet50_v1, {}, (10, 3, 32, 32)),
    (Resnet50_v1, {}, (1, 3, 244, 244)),
    (Resnet18_v2, {}, (100, 1, 28, 28)),
    (Resnet18_v2, {}, (100, 3, 32, 32)),
    (Resnet18_v2, {}, (2, 3, 244, 244)),
    (Resnet34_v2, {}, (100, 1, 28, 28)),
    (Resnet34_v2, {}, (100, 3, 32, 32)),
    (Resnet34_v2, {}, (2, 3, 244, 244)),
    (Resnet50_v2, {}, (10, 1, 28, 28)),
    (Resnet50_v2, {}, (10, 3, 32, 32)),
    (Resnet50_v2, {}, (1, 3, 244, 244)),
    (LeNet, {"lenet_quantized": True}, (100, 1, 28, 28)),
    (LeNet, {"lenet_quantized": True}, (100, 2, 28, 28)),
    (LeNet, {"lenet_quantized": True}, (100, 3, 28, 28)),
    (LeNet, {"lenet_quantized": True}, (100, 10, 28, 28)),
    (LeNet, {"lenet_quantized": False}, (100, 1, 28, 28)),
    (LeNet, {"lenet_quantized": False}, (100, 2, 28, 28)),
    (LeNet, {"lenet_quantized": False}, (100, 3, 28, 28)),
    (LeNet, {"lenet_quantized": False}, (100, 10, 28, 28)),
]


@pytest.mark.parametrize("model_class, kwargs, input_shape", TEST_INPUT_DATA)
def test_models(model_class, kwargs, input_shape) -> None:
    dataset = dataset_from_name("cifar10")
    assert dataset is CIFAR10
    dataset.shape = input_shape

    assert models_by_name[model_class.name] is model_class

    model = model_class(dataset=dataset, **kwargs)
    input_values = torch.Tensor(np.random.uniform(0, 1.0, input_shape))
    output = model(input_values)
    assert torch.equal(torch.as_tensor(output.shape), torch.Tensor([input_shape[0], dataset.num_classes]).long())
