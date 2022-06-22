from bitorch.datasets import MNIST, CIFAR10, CIFAR100, ImageNet

from bitorch.models import (
    models_by_name,
    LeNet,
    Resnet,
    Resnet18V1,
    Resnet34V1,
    Resnet50V1,
    Resnet18V2,
    Resnet34V2,
    Resnet50V2,
    ResnetE,
    ResnetE18,
    ResnetE34,
)
import torch
import numpy as np
import pytest
import itertools

ALL_DATASETS = [MNIST, CIFAR10, CIFAR100, ImageNet]
RGB_DATASETS = [CIFAR10, CIFAR100, ImageNet]

TEST_INPUT_DATA = [
    [Resnet, {"resnet_version": [1, 2], "resnet_num_layers": [18, 34, 50]}, ALL_DATASETS],
    [Resnet18V1, {}, ALL_DATASETS],
    [Resnet34V1, {}, ALL_DATASETS],
    [Resnet50V1, {}, ALL_DATASETS],
    [Resnet18V2, {}, ALL_DATASETS],
    [Resnet34V2, {}, ALL_DATASETS],
    [Resnet50V2, {}, ALL_DATASETS],
    [ResnetE, {"resnete_num_layers": [18, 34]}, RGB_DATASETS],
    [ResnetE18, {}, RGB_DATASETS],
    [ResnetE34, {}, RGB_DATASETS],
    [LeNet, {"lenet_version": [0, 1, 2, 3, 4]}, [MNIST]],
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
            assert torch.equal(
                torch.as_tensor(output.shape), torch.Tensor([input_shape[0], dataset.num_classes]).long()
            )
