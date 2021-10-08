from bitorch.datasets.mnist import MNIST
from bitorch.datasets.cifar import CIFAR10, CIFAR100
from bitorch.datasets.imagenet import ImageNet

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

ALL_DATASETS = [MNIST, CIFAR10, CIFAR100, ImageNet]
RGB_DATASETS = [CIFAR10, CIFAR100, ImageNet]

TEST_INPUT_DATA = [
    [Resnet, {"resnet_version": [1, 2], "resnet_num_layers": [18, 34, 50]}, ALL_DATASETS],
    [Resnet18_v1, {}, ALL_DATASETS],
    [Resnet34_v1, {}, ALL_DATASETS],
    [Resnet50_v1, {}, ALL_DATASETS],
    [Resnet18_v2, {}, ALL_DATASETS],
    [Resnet34_v2, {}, ALL_DATASETS],
    [Resnet50_v2, {}, ALL_DATASETS],
    [Resnet_E, {"resnete_num_layers": [18, 34]}, RGB_DATASETS],
    [Resnet_E18, {}, RGB_DATASETS],
    [Resnet_E34, {}, RGB_DATASETS],
    [LeNet, {"lenet_quantized": [True, False]}, [MNIST]],
]


@pytest.mark.parametrize("model_class, model_kwargs, datasets_to_test", TEST_INPUT_DATA)
@pytest.mark.parametrize("dataset", ALL_DATASETS)
def test_models(model_class, model_kwargs, datasets_to_test, dataset) -> None:
    assert models_by_name[model_class.name] is model_class
    if dataset not in datasets_to_test:
        pytest.skip(f"Model '{model_class.name}' does not need to work with the dataset '{dataset.name}'.")

    all_model_kwargs_combinations = [
        dict(zip(model_kwargs.keys(), combination)) for combination in itertools.product(*model_kwargs.values())
    ]

    for combination in all_model_kwargs_combinations:
        batch_sizes_to_test = [2, 10]
        if dataset is ImageNet:
            batch_sizes_to_test = [1, 2]
        for batch_size in batch_sizes_to_test:
            input_shape = list(dataset.shape)
            input_shape[0] = batch_size

            model = model_class(dataset=dataset, **combination)
            input_values = torch.Tensor(np.random.uniform(0, 1.0, input_shape))
            output = model(input_values)
            assert torch.equal(torch.as_tensor(output.shape), torch.Tensor(
                [input_shape[0], dataset.num_classes]).long())
