from typing import Any, Union, Optional
from torch import Tensor
from torch.nn import EmbeddingBag, Embedding
from torch.nn.functional import embedding_bag, embedding


from bitorch.layers.config import config
from bitorch.quantizations import Quantization


class QEmbeddingBag(EmbeddingBag):
    """Quantized version of pytorchs embedding bag. With the input indices the embedding is computed with a quantized
    version of the layers weight table. The output embedding will be also quantized before return.
    """

    def __init__(
        self,
        *args: Any,
        embedding_dim: int,
        weight_quantization: Optional[Union[Quantization, str]] = None,
        output_quantization: Optional[Union[Quantization, str]] = None,
        **kwargs: Any,
    ) -> None:
        super(QEmbeddingBag, self).__init__(*args, embedding_dim=embedding_dim, **kwargs)  # type: ignore
        """load quantization functions"""
        self.embedding_weight_quantization = config.get_quantization_function(
            weight_quantization or config.weight_quantization
        )
        self.embedding_input_quantization = config.get_quantization_function(
            output_quantization or config.input_quantization
        )

    def forward(
        self,
        input: Tensor,
        offsets: Optional[Tensor] = None,
        per_sample_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """generates embeddings for received bags. then quantizes these embeddings and depending on configuration
        forwards it through another quantized linear layer.

        Args:
            input (Tensor): indices list for embeddings
            offsets (Optional[Tensor], optional): offsets to determine embedding sequences. Defaults to None.
            per_sample_weights (Optional[Tensor], optional): sample weights. Defaults to None.

        Returns:
            Tensor: embeddings for given sequences
        """
        # necessary for torch 1.8 compliance
        if hasattr(self, "padding_idx"):
            embeddings = embedding_bag(
                input,
                self.embedding_weight_quantization(self.weight),
                offsets,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.mode,
                self.sparse,
                per_sample_weights,
                self.include_last_offset,
                self.padding_idx,
            )
        else:
            embeddings = embedding_bag(
                input,
                self.embedding_weight_quantization(self.weight),
                offsets,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.mode,
                self.sparse,
                per_sample_weights,
                self.include_last_offset,
            )
        embeddings = self.embedding_input_quantization(embeddings)
        return embeddings


class QEmbedding(Embedding):
    """Quantized version of pytorchs embedding layer. With input indices the embedding is computed with a quantized
    version of the layers weight table. The output embedding will be also quantized before return.
    """

    def __init__(
        self,
        *args: Any,
        embedding_dim: int,
        weight_quantization: Optional[Union[Quantization, str]] = None,
        output_quantization: Optional[Union[Quantization, str]] = None,
        **kwargs: Any,
    ) -> None:
        super(QEmbedding, self).__init__(*args, embedding_dim=embedding_dim, **kwargs)  # type: ignore
        """load quantization functions"""
        self.embedding_weight_quantization = config.get_quantization_function(
            weight_quantization or config.weight_quantization
        )
        self.embedding_output_quantization = config.get_quantization_function(
            output_quantization or config.input_quantization
        )

    def forward(self, input: Tensor) -> Tensor:
        """generates embeddings for received bags. then quantizes these embeddings and depending on configuration
        forwards it through another quantized linear layer.

        Args:
            input (Tensor): indices for embeddings

        Returns:
            Tensor: embeddings for given sequences
        """
        embeddings = embedding(
            input,
            self.embedding_weight_quantization(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        embeddings = self.embedding_output_quantization(embeddings)
        return embeddings
