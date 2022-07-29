import numpy as np
from sklearn import metrics
import logging
from argparse import Namespace
from typing import Union, Any, List

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.nn import Module, BCELoss
from torchmetrics import Accuracy, F1Score, Precision, Recall

from .unused_args import clean_hyperparameters
from .utils import create_optimizer, create_scheduler


class ModelWrapper(LightningModule):
    """Wrapper class for a pytorch model to fully utilize pytorch lightning functionality"""

    def __init__(
        self,
        model: Module,
        num_classes: int,
        args: Namespace,
        add_f1_prec_recall: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(clean_hyperparameters(args))
        self.loss_function = BCELoss()
        self.model = model
        self.train_accuracy_top1 = Accuracy(num_classes=num_classes)
        self.train_accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.accuracy_top1 = Accuracy(num_classes=num_classes)
        self.accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.add_f1_prec_recall = add_f1_prec_recall
        if add_f1_prec_recall:
            self.f1 = F1Score(num_classes=num_classes)
            self.prec = Precision(num_classes=num_classes)
            self.recall = Recall(num_classes=num_classes)

    def configure_optimizers(self) -> Union[dict, torch.optim.Optimizer]:  # type: ignore
        logging.info(f"Using {self.hparams.optimizer} optimizer and {self.hparams.lr_scheduler} lr scheduler...")
        optimizer = create_optimizer(self.hparams.optimizer, self.model, self.hparams.lr, self.hparams.momentum)
        if self.hparams.lr_scheduler is not None:
            scheduler = create_scheduler(
                self.hparams.lr_scheduler,
                optimizer,
                self.hparams.lr_factor,
                self.hparams.lr_steps,
                self.hparams.max_epochs,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # type: ignore
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
        y_hat = self.model(dense_values, (categorical_values_i, categorical_values_o))

        loss = self.loss_function(torch.squeeze(y_hat), torch.squeeze(y))
        self.log_dict({"loss/train": loss})
        return loss

    def validation_step_end(self, *args: Any, **kwargs: Any) -> Any:
        """calculate all them metrics and log via wandb/tensorboard"""
        y = torch.cat(list(map(lambda x: x["y"], self.validation_results)))
        y_hat = torch.cat(list(map(lambda x: x["y_hat"], self.validation_results)))
        loss = self.loss_function(y, y_hat)
        rmse = torch.sqrt(F.mse_loss(y_hat, y)).item()
        y_array = np.array(y.cpu())
        y_hat_array = np.array(y_hat.cpu()) >= 0.5
        balanced_accuracy = metrics.balanced_accuracy_score(y_array, y_hat_array)
        accuracy = metrics.accuracy_score(y_array, y_hat_array)
        f1 = metrics.f1_score(y_array, y_hat_array)
        roc_auc = metrics.roc_auc_score(y_array, y_hat.cpu())
        precision = metrics.precision_score(y_array, y_hat_array)
        recall = metrics.recall_score(y_array, y_hat_array)
        self.log_dict(
            {
                "val_los": loss,
                "val_rmse": rmse,
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "balanced accuracy": balanced_accuracy,
                "accuracy": accuracy,
                "f1 score": f1,
            },
            prog_bar=True,
        )
        return super().validation_step_end(*args, **kwargs)

    def on_validation_start(self) -> None:
        self.validation_results: List[dict] = []
        return super().on_validation_start()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # type: ignore
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
        y_hat = torch.squeeze(self.model(dense_values, (categorical_values_i, categorical_values_o)))
        y = torch.squeeze(y)
        y_hat = torch.squeeze(y_hat)
        self.validation_results.append({"y": y, "y_hat": y_hat})
