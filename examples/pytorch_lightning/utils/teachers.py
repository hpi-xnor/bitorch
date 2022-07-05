from typing import Dict, List

from torch import nn


from torchvision import models


def _teachers() -> Dict[str, nn.Module]:
    def resnet18():
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    def resnet34():
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    def resnet50_v1():
        # Old weights with accuracy 76.130%
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    def resnet50_v2():
        # New weights with accuracy 80.858%
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    def resnet50():
        # New weights with accuracy 80.858%
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    return locals()


def available_teachers() -> List[str]:
    return list(_teachers().keys())


def get_teacher(teacher_name: str) -> nn.Module:
    return _teachers()[teacher_name]()
