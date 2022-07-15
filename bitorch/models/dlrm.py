from argparse import ArgumentParser
from enum import Enum
from typing import List, Tuple, Union
import logging
import torch
from torch.nn import (
    Linear,
    Sequential,
    PReLU,
    Sigmoid,
    EmbeddingBag,
    ModuleList,
    BatchNorm1d,
)
import numpy as np
from bitorch.datasets.base import BasicDataset
from bitorch.layers import QLinear
from bitorch.models.base import Model

# from .utils import create_loss_function, create_optimizer, create_activation_function, parse_layer_sizes, str2bool
from bitorch.layers.qembedding import QEmbeddingBag


def parse_layer_sizes(layer_sizes_str: str) -> List[int]:
    layer_sizes_str = layer_sizes_str.replace('[', '').replace(']', '')
    return [int(size) for size in layer_sizes_str.split(",")]


class Interaction_Operation_Type(Enum):
    PRODUCT = "product"
    CONCAT = "concat"
    SUM = "sum"


def create_mlp(
        layer_sizes: List[int],
        quantized: bool = False) -> Sequential:
    """creates a mlp module

    Args:
        layer_sizes (List[int]): linear layer unit sizes
    for size in enumerate(layer_sizes_str.split(",")):
        parsed_layer_sizes.append(int(size))oid activation function.
            all other layers will have relu activation.
    """
    input_size = layer_sizes[0]
    mlp_layers = []

    for layer_size in layer_sizes[1:]:
        output_size = layer_size
        mlp_layers.append(BatchNorm1d(input_size))
        mlp_layers.append(
            QLinear(input_size, output_size, bias=False) if quantized else
            Linear(input_size, output_size, bias=True)
        )
        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (output_size + input_size))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        mlp_weight = np.random.normal(mean, std_dev, size=(output_size, input_size)).astype(np.float32)
        std_dev = np.sqrt(1 / output_size)  # np.sqrt(2 / (m + 1))
        mlp_bias = np.random.normal(mean, std_dev, size=output_size).astype(np.float32)
        # approach 1
        mlp_layers[-1].weight.data = torch.tensor(mlp_weight, requires_grad=True)
        if mlp_layers[-1].bias is not None:
            mlp_layers[-1].bias.data = torch.tensor(mlp_bias, requires_grad=True)

        mlp_layers.append(BatchNorm1d(output_size))
        mlp_layers.append(PReLU())
        input_size = output_size
    return Sequential(*mlp_layers)


def create_embeddings(
        embedding_dimension: int,
        layer_sizes: List[int],
        quantized: bool,
        sparse=False) -> Tuple[ModuleList, List[Union[None, torch.Tensor]]]:
    """creates the embedding layers for each category."""
    if sparse:
        logging.info("USING SPARSE EMBEDDINGS")
    embedding_layers = ModuleList()
    for layer_size in layer_sizes:
        logging.info(
            f"creating embedding layer with {layer_size} * {embedding_dimension} = {layer_size * embedding_dimension} params...")
        if quantized:
            embedding_layers.append(QEmbeddingBag(
                layer_size,
                embedding_dim=embedding_dimension,
                mode="mean",
                sparse=sparse,
            ))
        else:
            embedding_layers.append(EmbeddingBag(layer_size, embedding_dimension, mode="sum", sparse=sparse))
        embedding_weights = np.random.uniform(
            low=-np.sqrt(1 / layer_size), high=np.sqrt(1 / layer_size), size=(layer_size, embedding_dimension)
        ).astype(np.float32)
        embedding_layers[-1].weight.data = torch.tensor(embedding_weights, requires_grad=True)

    return embedding_layers


class DLRM(Model):
    name = "dlrm"
    total_size = 1.0
    inference_speed = 1.0
    validation_results = []

    def __init__(
            self,
            dataset: BasicDataset,
            dense_feature_size: int,
            embedding_dimension: int,
            embedding_layer_sizes: List[int],
            bottom_mlp_layer_sizes: List[int],
            top_mlp_layer_sizes: List[int],
            interaction_operation: Interaction_Operation_Type,
            binary_bottom_mlp: bool,
            binary_top_mlp: bool,
            binary_embedding: bool,
            **kwargs) -> None:
        super().__init__(dataset)
        self.interaction_operation = interaction_operation
        self.embedding_layers = create_embeddings(
            embedding_dimension,
            embedding_layer_sizes,
            binary_embedding,
        )

        bottom_mlp_layer_sizes = parse_layer_sizes(bottom_mlp_layer_sizes)
        top_mlp_layer_sizes = parse_layer_sizes(top_mlp_layer_sizes)

        # computing the correct bottom and top mlp layer sizes taking into account
        # feature dimensions and feature interaction output shapes
        bottom_mlp_layer_sizes = [dense_feature_size, *bottom_mlp_layer_sizes, embedding_dimension]

        if interaction_operation == Interaction_Operation_Type.CONCAT.value:
            top_mlp_layer_sizes = [(len(embedding_layer_sizes) + 1) * embedding_dimension, *top_mlp_layer_sizes]
        elif interaction_operation == Interaction_Operation_Type.PRODUCT.value:
            top_mlp_layer_sizes = [
                embedding_dimension + (len(embedding_layer_sizes) + 1) * ((len(embedding_layer_sizes) + 1) // 2), *top_mlp_layer_sizes]
        self.bottom_mlp = create_mlp(
            bottom_mlp_layer_sizes,
            quantized=binary_bottom_mlp,
        )
        self.top_mlp = create_mlp(
            top_mlp_layer_sizes,
            quantized=binary_top_mlp,
        )
        self.top_mlp[-1] = Sigmoid()

    @staticmethod
    def add_argparse_arguments(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("DLRM Model")
        parser.add_argument("--bottom-mlp-layer-sizes", type=str, default="[512, 256, 64]",
                            help="layer sizes of the bottom mlp")
        parser.add_argument("--top-mlp-layer-sizes", type=str, default="[512, 256, 1]",
                            help="layer sizes of the top mlp")
        parser.add_argument("--embedding-dimension", type=int, default=16,
                            help="number of embedding dimensions")
        parser.add_argument(
            "--interaction-operation",
            choices=[
                Interaction_Operation_Type.CONCAT.value,
                Interaction_Operation_Type.PRODUCT.value],
            default=Interaction_Operation_Type.CONCAT.value)
        parser.add_argument("--dense-embeddings", action="store_false", help="Disable sparse embeddings")

        parser.add_argument("--binary-embedding", action="store_true", default=False,
                            help="toggles use of binary embeddings in model.")
        parser.add_argument("--binary-top-mlp", action="store_true", default=False,
                            help="toggles use of binary top mlp in model.")
        parser.add_argument("--binary-bottom-mlp", action="store_true", default=False,
                            help="toggles use of binary bottom mlp in model.")
        return parent_parser

    def forward_embeddings(self, categorical_values_i: torch.Tensor,
                           categorical_values_o: torch.Tensor) -> List[torch.Tensor]:
        """forwards the preprocessed data through the embedding layers."""
        embedding_outputs = []
        for index, embedding_layer in enumerate(self.embedding_layers):
            index_group = categorical_values_i[index]
            offset_group = categorical_values_o[index]
            embedding_outputs.append(embedding_layer(index_group, offset_group))
        return embedding_outputs

    def feature_interaction(self, mlp_output: torch.Tensor, embedding_outputs: List[torch.Tensor]):
        if self.interaction_operation == Interaction_Operation_Type.PRODUCT.value:
            batch_size, dimension = mlp_output.shape
            concated_values = torch.cat([mlp_output] + embedding_outputs, dim=1).view((batch_size, -1, dimension))
            product_matrix = torch.bmm(concated_values, torch.transpose(concated_values, 1, 2))
            _, ni, nj = product_matrix.shape
            li = torch.tensor([i for i in range(ni) for j in range(i + 0)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + 0)])
            flat_product_matrix = product_matrix[:, li, lj]
            result = torch.cat([mlp_output, flat_product_matrix], dim=1)
        elif self.interaction_operation == Interaction_Operation_Type.CONCAT.value:
            result = torch.cat([mlp_output] + embedding_outputs, dim=1)
        else:
            raise ValueError("Interaction operation not supported!")

        return result

    def forward(self, dense_values, categorical_values):
        mlp_output = self.bottom_mlp(dense_values)
        embedding_outputs = self.forward_embeddings(*categorical_values)
        feature_interactions = self.feature_interaction(mlp_output, embedding_outputs)
        interaction_probability = self.top_mlp(feature_interactions)

        # if the top mlp has multiple output values, aggregate these into one single value
        if len(interaction_probability.shape) > 1 and interaction_probability.shape[1] > 1:
            interaction_probability = torch.clamp(interaction_probability, 0, 1)
            interaction_probability = torch.mean(interaction_probability, dim=1)
        return interaction_probability
