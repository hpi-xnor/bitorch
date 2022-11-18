from typing import Union, Optional, Dict, List, Tuple
import numpy
import torch
import warnings
from bitorch.layers.config import config
from bitorch.quantizations.base import Quantization
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class BEmbedding(nn.Module):
    """Binarized version of pytorchs embedding layer. Uses given binarization method to binarize the weights.
    Memory consumption during training increases with batch size. Inference is always small.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weight_quantization: Union[Quantization, str, None] = None,
        device: Union[str, torch.device, None] = None,
        sign_bool: bool = False,  # Whether a boolean 0 represents a -1. Set to True for Sign.
    ) -> None:
        super().__init__()
        # Load the quantization function
        self.weight_quantization = config.get_quantization_function(weight_quantization or config.weight_quantization)
        # Random initialize the weight. Can be set using set_weight.
        self.weight: Union[Parameter, Tensor] = Parameter(
            torch.rand((num_embeddings, embedding_dim), device=device) > 0.5, requires_grad=False
        )

        self.padding_idx = padding_idx
        self.sign_bool = sign_bool

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.optimizer_dict: Optional[Dict[str, List[Tensor]]] = None
        self.unique: Optional[Tensor] = None
        self.unique_vectors: Optional[Tensor] = None
        self.out_param: Optional[Tensor] = None

    def select_unique_vectors(self, flat_indices: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Given a flat tensor of indices, return the unique indices, their inverse in the original tensor,
        and a tensor with embedding vectors that are indexed by the unique indices.

        Args:
            flat_indices (Tensor): A flat tensor of indices that query the embedding table.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: unqiue indices, inverse indices, unique indexed embedding vectors
        """
        unique, inverse_indices = self.unique_wrapper(flat_indices)
        unique_weight = self.weight.index_select(0, unique).to(torch.float32)
        return unique, inverse_indices, unique_weight

    def unique_wrapper(self, tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the unique values and inverse indices of a given tensor. Uses numpy when on cpu and otherwise pytorch.

        Args:
            tensor (Tensor): Tensor to compute the unique values from.

        Returns:
            Tuple[Tensor, Tensor]: unique values, inverse indices
        """
        if tensor.device.type == "cpu":
            unique, inverse_indices = numpy.unique(tensor.numpy(), return_inverse=True)
            unique = torch.from_numpy(unique)
            inverse_indices = torch.from_numpy(inverse_indices)
        else:
            unique, inverse_indices = torch.unique(tensor, return_inverse=True)
        return unique, inverse_indices

    def apply_padding(self, indices: Tensor, embedding_vectors: Tensor) -> Tensor:
        """Applies padding to the embedding vectors. Sets the embedding vector to zero where
        the given unique index matches the padding_idx property. This operation is inplace.

        Args:
            indices (Tensor): Indices of the embedding vectors.
            embedding_vectors (Tensor): Embedding vectors to be padded.

        Returns:
            Tensor: Padded embedding vectors.
        """
        if self.padding_idx is not None:
            embedding_vectors[indices == self.padding_idx] = 0
        return embedding_vectors

    def transform_zeros(self, embedding_vectors: Tensor) -> Tensor:
        """If the sign_bool property is set, replaces 0 with -1. This operation is inplace.

        Args:
            embedding_vectors (Tensor): The tensor to be modified.

        Returns:
            Tensor: The modified input tensor
        """
        if self.sign_bool:
            embedding_vectors[embedding_vectors == 0] = -1
        return embedding_vectors

    def set_optimizable_weights(self, weights: Tensor) -> None:
        """Inject the weights to be optimized into the optimizer.

        Args:
            weights (Tensor): The weights to be ioptimized.
        """
        if self.optimizer is not None:
            if self.optimizer_dict is None:
                self.optimizer_dict = {"params": [weights]}
                self.optimizer.add_param_group(self.optimizer_dict)
            elif self.optimizer_dict:
                self.optimizer.state[weights] = self.optimizer.state[self.optimizer_dict["params"][0]]
                del self.optimizer.state[self.optimizer_dict["params"][0]]
            self.optimizer_dict["params"] = [weights]

    def forward(self, input: Tensor) -> Tensor:
        """Generates embeddings for received tokens.

        Args:
            input (Tensor): indices for embedding

        Returns:
            Tensor: embeddings for given token
        """
        input_shape = input.shape
        self.unique, inverse_indices, self.unique_vectors = self.select_unique_vectors(input.flatten())
        self.apply_padding(self.unique, self.unique_vectors)
        self.transform_zeros(self.unique_vectors)
        self.unique_vectors.requires_grad_(True)
        out = self.unique_vectors.index_select(0, inverse_indices)
        self.set_optimizable_weights(self.unique_vectors)
        return out.reshape((*input_shape, -1))

    def set_weight(self, weight: Tensor) -> None:
        if weight.dtype != torch.bool:
            weight = self.weight_quantization(weight) == 1
        self.weight.copy_(weight)

    @torch.no_grad()
    def step(self) -> None:
        """Step the BEmbedding by copying the optimized unique embedding vectors into the binary embedding table."""
        assert self.unique is not None and self.unique_vectors is not None, "Call forward before step."
        if self.padding_idx is not None:
            self.unique_vectors = self.unique_vectors[self.unique != self.padding_idx]
            self.unique = self.unique[self.unique != self.padding_idx]
        self.weight.index_copy_(0, self.unique, self.weight_quantization(self.unique_vectors) == 1)
        self.unique = None
        self.unique_vectors = None

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set the optimizer to set parameters to be optimized dynamically during training.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer of the `BEmbedding`.
        """
        self.optimizer = optimizer


class BEmbeddingBag(BEmbedding):
    """Binarized version of pytorchs embedding bag. Uses given binarization method to binarize the weights.
    Memory consumption during training increases with batch size. Inference is always small.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        weight_quantization: Union[Quantization, str, None] = None,
        device: Union[str, torch.device, None] = None,
        sign_bool: bool = False,  # Whether a boolean 0 represents a -1.
        mode: str = "mean",
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            weight_quantization=weight_quantization,
            device=device,
            sign_bool=sign_bool,
        )
        self.mode = mode
        self.embedding_dim = embedding_dim
        warnings.warn(
            "The BEmbeddingBag is experimental. Using the BEmbeddingBag leads to significal slowdowns of the model."
        )

    def apply_aggregate(
        self, batch_size: int, offsets: Tensor, inverse_indices: Tensor, unqiue_embedding_vectors: Tensor
    ) -> Tensor:
        """Aggregates the unique embedding vectors using the defined mode.

        Args:
            batch_size (int): Batch size of the input data.
            offsets (Tensor): Offsets of inverse indices for each batch. Defines which embedding vectors are aggregated.
            inverse_indices (Tensor): Flattened bag of indices for each batch.
            unqiue_embedding_vectors (Tensor): Unique embedding vectors to be aggregated.

        Returns:
            Tensor: The aggregated embedding vectors.
        """
        out = torch.zeros((batch_size, self.embedding_dim), device=self.weight.device)
        for row, (start_index, end_index) in enumerate(zip(offsets.tolist(), offsets.tolist()[1:] + [None])):
            use_indices = inverse_indices[start_index:end_index]
            if self.mode == "sum":
                out[row] = torch.sum(unqiue_embedding_vectors.index_select(0, use_indices), dim=0)
            elif self.mode == "mean":
                out[row] = torch.sum(unqiue_embedding_vectors.index_select(0, use_indices), dim=0).div_(
                    len(use_indices)
                )
            elif self.mode == "prod":
                out[row] = torch.prod(unqiue_embedding_vectors.index_select(0, use_indices), dim=0)
        return out.reshape((batch_size, -1))

    def forward(self, indices: Tensor, offsets: Tensor) -> Tensor:  # type: ignore
        """Generates embeddings from given tokens and offsets.

        Args:
            indices (Tensor): The tokens to be embedded.
            offsets (Tensor): The offsets describing the starting points of batch items.

        Returns:
            Tensor: The embedded and aggregated tokens.
        """
        self.unique, inverse_indices, self.unique_vectors = self.select_unique_vectors(indices.flatten())
        self.apply_padding(self.unique, self.unique_vectors)
        self.transform_zeros(self.unique_vectors)
        self.unique_vectors.requires_grad_(True)
        batch_size = offsets.size(0)
        out = self.apply_aggregate(
            batch_size=offsets.size(0),
            offsets=offsets,
            inverse_indices=inverse_indices,
            unqiue_embedding_vectors=self.unique_vectors,
        )
        self.set_optimizable_weights(self.unique_vectors)
        return out.reshape((batch_size, -1))
