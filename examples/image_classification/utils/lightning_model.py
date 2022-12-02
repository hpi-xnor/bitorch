# type: ignore
import logging
from argparse import Namespace
from typing import Union, Any

import torch
from pytorch_lightning import LightningModule
from torch.nn import Module, CrossEntropyLoss
from torchmetrics import Accuracy, F1Score, Precision, Recall

from .kd_loss import DistributionLoss
from .teachers import get_teacher
from .unused_args import clean_hyperparameters
from .utils import create_optimizer, create_scheduler


class ModelWrapper(LightningModule):
    def __init__(
        self,
        model: Module,
        num_classes: int,
        quantization_scheduler: Module,
        script_args: Namespace,
        add_f1_prec_recall: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(clean_hyperparameters(script_args))
        self.loss_function = CrossEntropyLoss()
        self.model = model
        self.batch_accuracy_top1 = Accuracy(num_classes=num_classes)
        self.batch_accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.train_accuracy_top1 = Accuracy(num_classes=num_classes)
        self.train_accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.accuracy_top1 = Accuracy(num_classes=num_classes)
        self.accuracy_top5 = Accuracy(top_k=5, num_classes=num_classes)
        self.add_f1_prec_recall = add_f1_prec_recall
        self.quantization_scheduler = quantization_scheduler
        if add_f1_prec_recall:
            self.f1 = F1Score(num_classes=num_classes)
            self.prec = Precision(num_classes=num_classes)
            self.recall = Recall(num_classes=num_classes)

    def training_step(self, batch: torch.Tensor) -> torch.Tensor:  # type: ignore
        x_train, y_train = batch

        y_hat = self.model(x_train)
        loss = self.calculate_loss(x_train, y_train, y_hat)

        self.batch_accuracy_top1(y_hat, y_train)
        self.batch_accuracy_top5(y_hat, y_train)
        self.train_accuracy_top1(y_hat, y_train)
        self.train_accuracy_top5(y_hat, y_train)

        self.log_dict(
            {
                "metrics/batch-top1-accuracy": self.batch_accuracy_top1,
                "metrics/batch-top5-accuracy": self.batch_accuracy_top5,
                "loss/train": loss,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log_dict(
            {
                "metrics/train-top1-accuracy": self.train_accuracy_top1,
                "metrics/train-top5-accuracy": self.train_accuracy_top5,
            },
            on_step=False,
            on_epoch=True,
        )
        return loss

    def calculate_loss(self, x_train: torch.Tensor, y_train: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.loss_function(y_hat, y_train)

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # type: ignore
        x_test, y_test = batch

        y_hat = self.model(x_test)
        loss = self.loss_function(y_hat, y_test)

        self.accuracy_top1(y_hat, y_test)
        self.accuracy_top5(y_hat, y_test)

        metrics_dict = {
            "metrics/test-top1-accuracy": self.accuracy_top1,
            "metrics/test-top5-accuracy": self.accuracy_top5,
            "loss/test": loss,
        }

        if self.add_f1_prec_recall:
            self.f1(y_hat, y_test)
            self.prec(y_hat, y_test)
            self.recall(y_hat, y_test)
            metrics_dict.update(
                {
                    "metrics/f1": self.f1,
                    "metrics/precision": self.prec,
                    "metrics/recall": self.recall,
                }
            )
        self.log_dict(metrics_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        if self.quantization_scheduler is not None:
            self.quantization_scheduler.step()
            self.log(
                "quantization_scheduler/factor",
                self.quantization_scheduler.scheduled_quantizer_instances[0].factor,
            )
        return super().on_validation_epoch_end()

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


class DistillationModelWrapper(ModelWrapper):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._kd_loss = DistributionLoss()
        logging.info(f"Training with Knowledge Distillation, loading teacher {self.hparams.teacher}.")
        self.teacher = get_teacher(self.hparams.teacher)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def calculate_loss(self, x_train: torch.Tensor, y_train: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        y_hat_teacher = self.teacher.forward(x_train)
        return self._kd_loss(y_hat, y_hat_teacher)
