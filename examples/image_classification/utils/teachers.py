from typing import Dict, List

from torch import nn


from torchvision import models


def _teachers() -> Dict[str, nn.Module]:
    def resnet18() -> nn.Module:
        return models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    def resnet34() -> nn.Module:
        return models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    def resnet50_v1() -> nn.Module:
        # Old weights with accuracy 76.130%
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    def resnet50_v2() -> nn.Module:
        # New weights with accuracy 80.858%
        return models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    def resnet50() -> nn.Module:
        # New weights with accuracy 80.858%
        return models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    return locals()


def available_teachers() -> List[str]:
    """Return a list of all available model names (pre-trained on ImageNet)."""
    return list(_teachers().keys())


def get_teacher(teacher_name: str) -> nn.Module:
    """Return a model pretrained on ImageNet for a given model name."""
    return _teachers()[teacher_name]()
