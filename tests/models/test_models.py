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
    DLRM,
    DenseNet28,
    DenseNet37,
    DenseNet45,
    DenseNetFlex,
    MeliusNet22,
    MeliusNet42,
    MeliusNetFlex,
    MeliusNet23,
    MeliusNet59,
    MeliusNetA,
    MeliusNetB,
    MeliusNetC,
    QuickNet,
    QuickNetSmall,
    QuickNetLarge,
)
import torch
import numpy as np
import pytest
import itertools


MNIST = [(1, 1, 28, 28), 10, "MNIST"]
CIFAR10 = [(1, 3, 32, 32), 10, "CIFAR10"]
CIFAR100 = [(1, 3, 32, 32), 100, "CIFAR100"]
IMAGENET = [(1, 3, 224, 224), 1000, "IMAGENET"]

CRITEO = [([1, 13], ([26, 1], [26, 1])), 1, "CRITEO"]

ALL_IMAGE_DATASETS = [MNIST, CIFAR10, CIFAR100, IMAGENET]
RGB_DATASETS = [CIFAR10, CIFAR100, IMAGENET]

TEST_INPUT_DATA = [
    [
        Resnet,
        {"resnet_version": [1, 2], "resnet_num_layers": [18, 34, 50]},
        ALL_IMAGE_DATASETS,
    ],
    [Resnet18V1, {}, ALL_IMAGE_DATASETS],
    [Resnet34V1, {}, ALL_IMAGE_DATASETS],
    [Resnet50V1, {}, ALL_IMAGE_DATASETS],
    [Resnet18V2, {}, ALL_IMAGE_DATASETS],
    [Resnet34V2, {}, ALL_IMAGE_DATASETS],
    [Resnet50V2, {}, ALL_IMAGE_DATASETS],
    [DenseNet28, {}, ALL_IMAGE_DATASETS],
    [DenseNet37, {}, ALL_IMAGE_DATASETS],
    [DenseNet45, {}, ALL_IMAGE_DATASETS],
    [DenseNetFlex, {"flex_block_config": [[6, 6, 6, 5]]}, ALL_IMAGE_DATASETS],
    [MeliusNet22, {}, ALL_IMAGE_DATASETS],
    [MeliusNet23, {}, ALL_IMAGE_DATASETS],
    [MeliusNet42, {}, ALL_IMAGE_DATASETS],
    [MeliusNet59, {}, ALL_IMAGE_DATASETS],
    [MeliusNetA, {}, ALL_IMAGE_DATASETS],
    [MeliusNetB, {}, ALL_IMAGE_DATASETS],
    [MeliusNetC, {}, ALL_IMAGE_DATASETS],
    [MeliusNetFlex, {"flex_block_config": [[6, 6, 6, 5]]}, ALL_IMAGE_DATASETS],
    [ResnetE, {"resnete_num_layers": [18, 34]}, RGB_DATASETS],
    [ResnetE18, {}, RGB_DATASETS],
    [ResnetE34, {}, RGB_DATASETS],
    [LeNet, {"lenet_version": [0, 1, 2, 3, 4]}, [MNIST]],
    [DLRM, {}, [CRITEO]],
    [QuickNet, {}, [IMAGENET]],
    [QuickNetSmall, {}, [IMAGENET]],
    [QuickNetLarge, {}, [IMAGENET]],
]


@pytest.mark.parametrize("model_class, model_kwargs, datasets_to_test", TEST_INPUT_DATA)
@pytest.mark.parametrize("dataset", [MNIST, CIFAR10, CIFAR100, IMAGENET, CRITEO])
def test_models(model_class, model_kwargs, datasets_to_test, dataset) -> None:
    assert models_by_name[model_class.name.lower()] is model_class
    if dataset not in datasets_to_test:
        pytest.skip(f"Model '{model_class.name}' does not need to work with the dataset '{dataset[2]}'.")

    all_model_kwargs_combinations = [
        dict(zip(model_kwargs.keys(), combination)) for combination in itertools.product(*model_kwargs.values())
    ]

    for combination in all_model_kwargs_combinations:
        batch_sizes_to_test = [2, 10]
        if dataset is IMAGENET:
            batch_sizes_to_test = [1, 2]
        for batch_size in batch_sizes_to_test:
            if model_class.name == "DLRM":
                model = model_class(
                    dense_feature_size=dataset[0][0][1],
                    embedding_layer_sizes=[100] * dataset[0][1][0][0],
                    **combination,
                )
                dataset[0][0][0] = batch_size
                dataset[0][1][0][1] = batch_size
                dataset[0][1][1][1] = batch_size
                input_values = (
                    torch.Tensor(np.random.uniform(0, 1.0, dataset[0][0])),
                    (torch.zeros(dataset[0][1][0], dtype=int), torch.zeros(dataset[0][1][1], dtype=int)),
                )
                output = model(*input_values)
            else:
                input_shape = list(dataset[0])
                input_shape[0] = batch_size
                model = model_class(input_shape=dataset[0], num_classes=dataset[1], **combination)
                input_values = torch.Tensor(np.random.uniform(0, 1.0, input_shape))
                output = model(input_values)
            assert torch.equal(
                torch.as_tensor(output.shape),
                torch.Tensor([batch_size, dataset[1]]).long(),
            )
