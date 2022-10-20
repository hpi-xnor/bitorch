from typing import Tuple, Any

from torch.nn import Module

from bitorch.layers import QConv2d_NoAct, QConv1d_NoAct, QConv3d_NoAct, QLinearBase, QEmbeddingBag, QEmbedding


class WeightClipper(object):
    """
    Callable class for clipping weights of quantized or other layers.
    """

    qlayers = (QConv1d_NoAct, QConv2d_NoAct, QConv3d_NoAct, QLinearBase, QEmbeddingBag, QEmbedding)

    def __init__(self, clip_value: float = 1.0, layers: Tuple[Any, ...] = qlayers):
        self.clip_value = clip_value
        self.layers = layers

    def __call__(self, module: Module) -> None:
        if isinstance(module, self.layers):
            weights = module.weight.data
            weights = weights.clamp(-self.clip_value, self.clip_value)  # type: ignore
            module.weight.data = weights
