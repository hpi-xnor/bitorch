from typing import Union
import torch
from pytorch_lightning import LightningModule
from torch.nn import Module, CrossEntropyLoss
from utils.utils import create_optimizer, create_scheduler
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
import logging


class ModelWrapper(LightningModule):
    def __init__(
            self,
            model: Module,
            optimizer: str,
            lr: float,
            momentum: float,
            lr_scheduler: str,
            lr_factor: float,
            lr_steps: list,
            num_classes: int,
            epochs: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.loss_function = CrossEntropyLoss()
        self.model = model
        self.accuracy_top1 = Accuracy(num_classes=num_classes)
        self.accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.f1 = F1Score(num_classes=num_classes)
        self.prec = Precision(num_classes=num_classes)
        self.recall = Recall(num_classes=num_classes)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        x_train, y_train = batch

        y_hat = self.model(x_train)
        loss = self.loss_function(y_hat, y_train)
        self.log_dict({
            "loss/train": loss,
        })
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # type: ignore
        x_test, y_test = batch

        y_hat = self.model(x_test)
        loss = self.loss_function(y_hat, y_test)

        self.log_dict({
            "metrics/top1 accuracy": self.accuracy_top1(y_hat, y_test),
            "metrics/top5 accuracy": self.accuracy_top5(y_hat, y_test),
            "metrics/f1": self.f1(y_hat, y_test),
            "metrics/precision": self.prec(y_hat, y_test),
            "metrics/recall": self.recall(y_hat, y_test),
            "loss/test": loss,
        }, prog_bar=True)

    def configure_optimizers(self) -> Union[dict, torch.optim.Optimizer]:  # type: ignore
        logging.info(f"Using {self.hparams.optimizer} optimizer and {self.hparams.lr_scheduler} lr schedluer...")
        optimizer = create_optimizer(self.hparams.optimizer, self.model, self.hparams.lr, self.hparams.momentum)
        if self.hparams.lr_scheduler is not None:
            scheduler = create_scheduler(
                self.hparams.lr_scheduler, optimizer, self.hparams.lr_factor,
                self.hparams.lr_steps, self.hparams.epochs
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler
            }
        else:
            return optimizer
