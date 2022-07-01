from argparse import ArgumentParser
import math
from pytorch_lightning import LightningModule
from enum import Enum
from typing import List, Optional, Tuple, Union
import logging
import torch
from torch.nn import (
    Linear,
    Sigmoid,
    EmbeddingBag,
    ModuleList,
    ParameterList,
    Parameter,
    BatchNorm1d,
)
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as metrics
from bitorch.layers import QLinear

# from .utils import create_loss_function, create_optimizer, create_activation_function, parse_layer_sizes, str2bool
from bitorch.layers.qembedding import QEmbeddingBag

class Weighted_Pooling_Type(Enum):
    NONE = 0
    FIXED = 1
    LEARNED = 2

class Interaction_Operation_Type(Enum):
    PRODUCT = "product"
    CONCAT = "concat"
    SUM = "sum"

class MLP(torch.nn.Module):
    """Mlp class for DLRM mlps. this mainly is used to properly pass shortcut data"""
    name = "MLP"
    def __init__(self, layers: ModuleList, name: str = "MLP", shortcut_out_index: Union[int, None] = None, shortcut_in_index: Union[int, None] = None):
        super().__init__()
        self.shortcut_out_index = shortcut_out_index
        self.shortcut_in_index = shortcut_in_index
        self.layers = layers
        self.name = name
        logging.info(f"shortcut_out_index {shortcut_out_index}")
        logging.info(f"shortcut_in_index {shortcut_in_index}")

    def forward(self, X, shortcut_value = None):
        out = None
        Y = X
        for index, layer in enumerate(self.layers):
            if index == self.shortcut_in_index:
                Y = Y + shortcut_value
            Y = layer(Y)
            if index == self.shortcut_out_index:
                out = Y
        return Y, out
    
    def __str__(self) -> str:
        out = f"{self.name}\n"
        out += "="*10
        out += "\n".join(self.layers)
        return out

def create_mlp(
    layer_sizes: List[int],
    quantized: bool = False) -> MLP:
    """creates a mlp module

    Args:
        layer_sizes (List[int]): linear layer unit sizes
        sigmoid_layer_idx (int): the layer number to use a sigmoid activation function.
            all other layers will have relu activation.
    """
    input_size = layer_sizes[0]
    this_shortcut_out_index = None
    this_shortcut_in_index = None
    mlp_layers = [] if not batch_norm else [BatchNorm1d(input_size)]
    if full_precision_layers is None:
        full_precision_layers = [False] * len(layer_sizes)

    for idx, layer_size in enumerate(layer_sizes[1:]):
        output_size = layer_size
        if batch_norm_before_sign:
            mlp_layers.append(BatchNorm1d(input_size))
        mlp_layers.append(
            Linear(input_size, output_size, bias=True) if not quantized or full_precision_layers[idx + 1] else
            QLinear(input_size, output_size, bias=False)
        )

        if idx == shortcut_out_index and this_shortcut_out_index is None:
            this_shortcut_out_index = len(mlp_layers)
        if idx == shortcut_in_index and this_shortcut_in_index is None:
            this_shortcut_in_index = len(mlp_layers)
        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (output_size + input_size))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        mlp_weight = np.random.normal(mean, std_dev, size=(output_size, input_size)).astype(np.float32)
        std_dev = np.sqrt(1 / output_size)  # np.sqrt(2 / (m + 1))
        mlp_bias = np.random.normal(mean, std_dev, size=output_size).astype(np.float32)
        # approach 1
        mlp_layers[-1].weight.data = torch.tensor(mlp_weight, requires_grad=True)
        if mlp_layers[-1].bias is not None:
            mlp_layers[-1].bias.data = torch.tensor(mlp_bias, requires_grad=True)

        if batch_norm_before_relu:
            mlp_layers.append(BatchNorm1d(output_size))
        mlp_layers.append(Sigmoid() if idx == sigmoid_layer_idx else (create_activation_function(activation_function, bitwidth)))
        input_size = output_size
    if sigmoid_layer_idx == -1:
        mlp_layers[-1] = Sigmoid()
    return MLP(ModuleList(mlp_layers), name=name, shortcut_out_index=this_shortcut_out_index, shortcut_in_index=this_shortcut_in_index)

def create_embeddings(
        embedding_dimension: int,
        layer_sizes: List[int],
        weighted_pooling: Weighted_Pooling_Type,
        binary_embedding: bool,
        add_linear_to_binary_embeddings: bool,
        sparse=False) -> Tuple[ModuleList, List[Union[None, torch.Tensor]]]:
    """creates the embedding layers for each category."""
    if sparse:
        logging.info("USING SPARSE EMBEDDINGS")
    embedding_layers = ModuleList()
    weighted_layers = []
    for layer_size in layer_sizes:
        logging.info(f"creating embedding layer with {layer_size} * {embedding_dimension} = {layer_size * embedding_dimension} params...")
        if binary_embedding:
            embedding_layers.append(QEmbeddingBag(
                layer_size,
                embedding_dim=embedding_dimension,
                mode="mean",
                sparse=sparse,
                use_linear_layer=add_linear_to_binary_embeddings
            ))
        else:
            embedding_layers.append(EmbeddingBag(layer_size, embedding_dimension, mode="sum", sparse=sparse))
        embedding_weights = np.random.uniform(
                low=-np.sqrt(1 / layer_size), high=np.sqrt(1 / layer_size), size=(layer_size, embedding_dimension)
            ).astype(np.float32)
        embedding_layers[-1].weight.data = torch.tensor(embedding_weights, requires_grad=True)
        weighted_layers.append(
            torch.ones(layer_size, dtype=torch.float32)
            if not weighted_pooling == Weighted_Pooling_Type.NONE.value
            else None)
    
    return embedding_layers, weighted_layers


class DLRM(Module):
    total_size = 1.0
    inference_speed = 1.0
    validation_results = []

    def __init__(
            self,
            dense_feature_size: int,
            weighted_pooling: Weighted_Pooling_Type,
            embedding_dimension: int,
            embedding_layer_sizes: List[int],
            bottom_mlp_layer_sizes: List[int],
            bottom_sigmoid_layer_idx: int,
            top_mlp_layer_sizes: List[int],
            top_full_precision_layers: List[int],
            bottom_full_precision_layers: List[int],
            top_sigmoid_layer_idx: int,
            interaction_operation: Interaction_Operation_Type,
            binary_bottom_mlp: bool,
            binary_top_mlp: bool,
            batch_norm_before_relu: bool,
            batch_norm_before_sign: bool,
            binary_embedding: bool,
            add_linear_to_binary_embeddings: bool,
            activation_function: str,
            bitwidth: int,
            optimizer: str,
            momentum: float,
            lr: float,
            lr_scheduler: str,
            lr_factor: float,
            lr_steps: List[int],
            epochs: int,
            loss: str,
            loss_weights: Union[None, List[float]],
            threshold: float,
            shortcut: str = "none",
            **kwargs) -> None:
        super().__init__()
        self.interaction_operation = interaction_operation
        self.embedding_layers = create_embeddings(
            embedding_dimension,
            embedding_layer_sizes,
            binary_embedding,
        )

        bottom_mlp_layer_sizes, self.sc_out_index = parse_layer_sizes(bottom_mlp_layer_sizes)
        top_mlp_layer_sizes, self.sc_in_index = parse_layer_sizes(top_mlp_layer_sizes)

        # computing the correct bottom and top mlp layer sizes taking into account feature dimensions and feature interaction output shapes
        bottom_mlp_layer_sizes = [dense_feature_size, *bottom_mlp_layer_sizes, embedding_dimension]

        if interaction_operation == Interaction_Operation_Type.CONCAT.value:
            top_mlp_layer_sizes = [(len(embedding_layer_sizes) + 1) * embedding_dimension, *top_mlp_layer_sizes]
        elif interaction_operation == Interaction_Operation_Type.PRODUCT.value:
            top_mlp_layer_sizes = [embedding_dimension + (len(embedding_layer_sizes) + 1) * ((len(embedding_layer_sizes) + 1) // 2), *top_mlp_layer_sizes]

        self.bottom_mlp = create_mlp(
            bottom_mlp_layer_sizes,
            quantized=binary_bottom_mlp,
        )
        self.top_mlp = create_mlp(
            top_mlp_layer_sizes,
            quantized=binary_top_mlp,
        )
        self.loss_function, self.loss_weights = create_loss_function(loss, loss_weights)

    @staticmethod
    def add_argparse_arguments(parent_parser: ArgumentParser):
        parser = parent_parser.add_argument_group("DLRM Model")
        parser.add_argument("--bottom-mlp-layer-sizes", type=str, default="[512, 256, 64, 1]",
            help="layer sizes of the bottom mlp")
        parser.add_argument("--top-mlp-layer-sizes", type=str, default="[512, 256]",
        # parser.add_argument("--top-mlp-layer-sizes", type=int, nargs="*", default=[48, 512, 256, 1],
            help="layer sizes of the top mlp")
        parser.add_argument("--bottom-sigmoid-layer-idx", type=int, default=None,
            help="index of the sigmoid activation function in the bottom mlp (default is disabled, -1 is last)")
        parser.add_argument("--top-sigmoid-layer-idx", type=int, default=-1,
            help="index of the sigmoid activation function in the top mlp (None is disabled, -1 is last)")
        parser.add_argument("--embedding-dimension", type=int, default=16,
            help="number of embedding dimensions")
        parser.add_argument("--interaction-operation", choices=[Interaction_Operation_Type.CONCAT.value, Interaction_Operation_Type.PRODUCT.value], default=Interaction_Operation_Type.CONCAT.value)
        parser.add_argument("--weighted-pooling", choices=list(Weighted_Pooling_Type), default=Weighted_Pooling_Type.NONE.value)
        parser.add_argument("--loss", type=str, default="mse",
                     help="name of loss function")
        parser.add_argument('--loss-weights', nargs="*", default=None,
                            help='list loss weights. this is only used by bce loss')
        parser.add_argument("--lr-scheduler", type=str,
                           choices=["cosine", "step", "exponential"],
                           help="name of the lr scheduler to use. default to none")
        parser.add_argument("--lr", type=float, default=0.1,
                            help="initial learning rate (default: 0.1)")
        parser.add_argument('--lr-factor', default=0.1, type=float,
                            help='learning rate decay ratio. this is used only by the step and exponential lr scheduler')
        parser.add_argument('--lr-steps', nargs="*", default=[30, 60, 90],
                            help='list of learning rate decay epochs as list. this is used only by the step scheduler')
        parser.add_argument('--momentum', type=float, default=0.9,
                            help='momentum value for optimizer, default is 0.9. only used for sgd optimizer')
        parser.add_argument('--optimizer', type=str, default="sgd", choices=["adam", "sgd", "sparse_adam", "radam"],
                            help='the optimizer to use. default is adam.')
        parser.add_argument("--threshold", type=float, default=0.5, help="threshold which is used to binarize predictions")
        parser.add_argument("--lr-num-warmup-steps", type=int, default=0, help="number of warmups steps of the lr scheduler")
        parser.add_argument("--lr-decay-start-step", type=int, default=0, help="number of steps until the lr decays")
        parser.add_argument("--lr-num-decay-steps", type=int, default=0, help="number of steps how long the lr decays")
        parser.add_argument("--full-embeddings", action="store_false", help="Disable sparse embeddings")

        parser.add_argument("--add-linear-to-binary-embeddings", action="store", type=str2bool, default=False, 
            help="whether to add an linear layer to binary embeddings")
        parser.add_argument("--binary-embedding", action="store", type=str2bool, default=False,
                        help="toggles use of binary embeddings in model.")
        parser.add_argument("--binary-top-mlp", action="store", type=str2bool, default=False,
                        help="toggles use of binary top mlp in model.")
        parser.add_argument("--binary-bottom-mlp", action="store", type=str2bool, default=False,
                        help="toggles use of binary bottom mlp in model.")
        parser.add_argument("--batch-norm-before-relu", action="store", type=str2bool, default=False,
                        help="toggles use of binary bottom mlp in model.")
        parser.add_argument("--batch-norm-before-sign", action="store", type=str2bool, default=False,
                        help="toggles use of binary bottom mlp in model.")
        parser.add_argument("--activation-function", choices=["relu", "prelu", "pact"], default="relu", type=str, help="select activation function")
        parser.add_argument('--top-full-precision-layers', nargs="*", default=[],
                            help='list of learning rate decay epochs as list. this is used only by the step scheduler')
        parser.add_argument('--bottom-full-precision-layers', nargs="*", default=[],
                            help='list of learning rate decay epochs as list. this is used only by the step scheduler')
        return parent_parser

    def forward_embeddings(self, categorical_values_i: torch.Tensor, categorical_values_o: torch.Tensor) -> List[torch.Tensor]:
        """forwards the preprocessed data through the embedding layers."""
        embedding_outputs = []
        for index, embedding_layer in enumerate(self.embedding_layers):
            weight_pooling_layer = self.weight_pooling_layers[index]
            index_group = categorical_values_i[index]
            offset_group = categorical_values_o[index]
            if weight_pooling_layer is not None:
                per_sample_weights = weight_pooling_layer.gather(0, index_group)
            else:
                per_sample_weights = None
            embedding_outputs.append(embedding_layer(index_group, offset_group, per_sample_weights=per_sample_weights))
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
        mlp_output, shortcout_output = self.bottom_mlp(dense_values)
        embedding_outputs = self.forward_embeddings(*categorical_values)
        feature_interactions = self.feature_interaction(mlp_output, embedding_outputs)
        interaction_probability, _ = self.top_mlp(feature_interactions, shortcout_output)

        # if the top mlp has multiple output values, aggregate these into one single value
        if len(interaction_probability.shape) > 1 and interaction_probability.shape[1] > 1:
            interaction_probability = torch.clamp(interaction_probability, 0, 1)
            interaction_probability = torch.mean(interaction_probability, dim=1)
        return interaction_probability

    def training_step(self, batch, batch_idx):
        dense_values, categorical_values_i, categorical_values_o, y = batch
        if isinstance(categorical_values_i, list):
            for el in categorical_values_i:
                el.to(self.device)
        else:
            categorical_values_i = categorical_values_i.to(self.device)
        if isinstance(categorical_values_o, list):
            for el in categorical_values_o:
                el.to(self.device)
        else:
            categorical_values_o = categorical_values_o.to(self.device)
        dense_values.to(self.device)
        y_hat = self(dense_values, (categorical_values_i, categorical_values_o))

        loss = self.loss_function(torch.squeeze(y_hat), torch.squeeze(y))
        self.log_dict({ "loss": loss })
        return loss
    
    def validation_step_end(self, *args, **kwargs):
        """calculate all them metrics and log via wandb/tensorboard"""

        y = torch.cat(list(map(lambda x: x["y"], self.validation_results)))
        y_hat = torch.cat(list(map(lambda x: x["y_hat"], self.validation_results)))
        loss = self.loss_function(y, y_hat)
        rmse = torch.sqrt(F.mse_loss(y_hat, y)).item()
        y_array = np.array(y.cpu())
        y_hat_array = np.array(y_hat.cpu()) >= self.hparams.threshold
        balanced_accuracy = metrics.balanced_accuracy_score(y_array, y_hat_array)
        accuracy = metrics.accuracy_score(y_array, y_hat_array)
        f1 = metrics.f1_score(y_array, y_hat_array)
        roc_auc = metrics.roc_auc_score(y_array, y_hat.cpu())
        precision = metrics.precision_score(y_array, y_hat_array)
        recall = metrics.recall_score(y_array, y_hat_array)
        self.log_dict({
            "val_los": loss,
            "val_rmse": rmse,
            "roc_auc": roc_auc,
            "precision": precision,
            "recall": recall,
            "balanced accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "f1 score": f1,
            "model size": self.total_size,
            "inference speed": self.inference_speed,
            "weighted accuracy": accuracy / (self.total_size * self.inference_speed),
            "weighted speed accuracy": accuracy / (self.total_size * self.inference_speed),
            "log2 weighted speed accuracy": accuracy / math.log(self.total_size * self.inference_speed, 2.0)
        }, prog_bar=True)
        return super().validation_step_end(*args, **kwargs)

    def on_validation_start(self) -> None:
        self.validation_results = []
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        dense_values, categorical_values_i, categorical_values_o, y = batch
        dense_values = dense_values.to(self.device)
        if isinstance(categorical_values_i, list):
            for el in categorical_values_i:
                el.to(self.device)
        else:
            categorical_values_i = categorical_values_i.to(self.device)
        if isinstance(categorical_values_o, list):
            for el in categorical_values_o:
                el.to(self.device)
        else:
            categorical_values_o = categorical_values_o.to(self.device)
        y_hat = torch.squeeze(self(dense_values, (categorical_values_i, categorical_values_o)))
        y = torch.squeeze(y)
        y_hat = torch.squeeze(y_hat)
        self.validation_results.append({ "y": y, "y_hat": y_hat })

    def configure_optimizers(self):
        configuration = {}
        optimizer = create_optimizer(self.hparams.optimizer, self, self.hparams.lr, self.hparams.momentum)
        configuration["optimizer"] = optimizer
        lr_scheduler = LRPolicyScheduler(
            optimizer,
            self.hparams.lr_num_warmup_steps,
            self.hparams.lr_decay_start_step,
            self.hparams.lr_num_decay_steps,
        )
        if lr_scheduler != None:
            configuration["lr_scheduler"] = lr_scheduler_config = {
                # REQUIRED: The scheduler instance
                "scheduler": lr_scheduler,
                # The unit of the scheduler's step size, could also be 'step'.
                # 'epoch' updates the scheduler on epoch end whereas 'step'
                # updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                "monitor": "balanced accuracy",
                # If set to `True`, will enforce that the value specified 'monitor'
                # is available when the scheduler is updated, thus stopping
                # training if not found. If set to `False`, it will only produce a warning
                "strict": True,
                # If using the `LearningRateMonitor` callback to monitor the
                # learning rate progress, this keyword can be used to specify
                # a custom logged name
                "name": "LRScheduler",
            }
        return configuration

    def transform_categorical(self, categorical: List[torch.Tensor]):
        categorical_batch_indices = [[torch.tensor(*np.where(category_batch[batch_num].numpy())) for batch_num in range(category_batch.size(0))] for category_batch in categorical]
        offsets = [torch.tensor([len(index_array) for index_array in category_batch]) for category_batch in categorical_batch_indices]
        for category_index, category_sizes in enumerate(offsets):
            offsets[category_index] = offsets[category_index].roll(1)
            offsets[category_index][0] = 0
            for size_index in range(1, len(category_sizes)):
                offsets[category_index][size_index] += offsets[category_index][size_index - 1]

        concatenated_categorical_batch_indices = [torch.cat(category_batch) for category_batch in categorical_batch_indices]
        return list(zip(concatenated_categorical_batch_indices, offsets))
