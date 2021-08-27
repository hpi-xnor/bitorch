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
    Resnet50_v2,
    Resnet_E,
    Resnet_E18,
    Resnet_E34
)
import torch
import numpy as np
import pytest
import itertools

mnist_100 = [100, 1, 28, 28]
mnist_10 = [10, 1, 28, 28]
cifar_100 = [100, 3, 32, 32]
cifar_10 = [10, 3, 32, 32]
imagenet_2 = [2, 3, 224, 224]
imagenet_1 = [1, 3, 224, 224]
more_channels_mnist_100 = [100, 10, 28, 28]

TEST_INPUT_DATA = [
    [Resnet, {"resnet_version": [1, 2], "resnet_num_layers": [18, 34, 50]}, [mnist_10, mnist_100, cifar_10]],
    [Resnet18_v1, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [Resnet34_v1, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [Resnet50_v1, {}, [mnist_10, cifar_10, imagenet_1]],
    [Resnet18_v2, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [Resnet34_v2, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [Resnet50_v2, {}, [mnist_10, cifar_10, imagenet_1]],
    [Resnet_E, {"resnete_num_layers": [18, 34]}, [mnist_10, mnist_100, cifar_10]],
    [Resnet_E18, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [Resnet_E34, {}, [mnist_10, mnist_100, cifar_10, cifar_100, imagenet_1, imagenet_2]],
    [LeNet, {"lenet_quantized": [True, False]}, [mnist_10, mnist_100, more_channels_mnist_100]]
]


@pytest.mark.parametrize("model_class, kwargs, input_shapes", TEST_INPUT_DATA)
def test_models(model_class, kwargs, input_shapes) -> None:
    dataset = dataset_from_name("cifar10")
    assert dataset is CIFAR10
    assert models_by_name[model_class.name] is model_class

    kwarg_combinations = [dict(zip(kwargs.keys(), combination)) for combination in itertools.product(*kwargs.values())]

    for combination in kwarg_combinations:
        for input_shape in input_shapes:
            dataset.shape = input_shape
            model = model_class(dataset=dataset, **combination)
            input_values = torch.Tensor(np.random.uniform(0, 1.0, input_shape))
            output = model(input_values)
            assert torch.equal(torch.as_tensor(output.shape), torch.Tensor(
                [input_shape[0], dataset.num_classes]).long())
