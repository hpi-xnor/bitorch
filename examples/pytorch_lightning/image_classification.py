import os

if os.environ.get("REMOTE_PYCHARM_DEBUG_SESSION", False):
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        "localhost",
        port=int(os.environ.get("REMOTE_PYCHARM_DEBUG_PORT", "12345")),
        stdoutToServer=True,
        stderrToServer=True,
    )

import argparse
import logging
from pathlib import Path
from typing import List, Any, Type

import fvbitcore.nn as fv_nn
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, LightningLoggerBase
from torch.utils.data import DataLoader

import bitorch
from bitorch import apply_args_to_configuration, RuntimeMode
from bitorch.datasets import dataset_from_name
from bitorch.datasets.base import Augmentation
from bitorch.models import model_from_name
from bitorch.quantizations import Quantization
from examples.pytorch_lightning.utils.callbacks import ProgressiveSignScalerCallback
from examples.pytorch_lightning.utils.log import CommandLineLogger
from examples.pytorch_lightning.utils.wandb_logger import CustomWandbLogger
from utils.arg_parser import create_argparser
from utils.lightning_model import ModelWrapper, DistillationModelWrapper
from utils.utils import configure_logging

logger = logging.getLogger()


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset.

    Args:
        args (argparse.Namespace): cli arguments
        model_args (argparse.Namespace): model specific cli arguments
    """
    configure_logging(logger, args.log_file, args.log_level, args.log_stdout)

    # switch to RAW bitorch mode for distributed data parallel training
    bitorch.mode = RuntimeMode.RAW

    apply_args_to_configuration(args)

    output_dir = Path(args.result_directory)
    output_dir.mkdir(exist_ok=True)

    loggers: List[LightningLoggerBase] = []
    if args.tensorboard_log:
        loggers.append(TensorBoardLogger(str(output_dir), name="tensorboard"))
    if args.csv_log:
        loggers.append(CSVLogger(str(output_dir), name="csv"))
    if args.wandb_log:
        loggers.append(
            CustomWandbLogger(
                args,
                project=args.wandb_project,
                name=args.wandb_experiment,
                save_dir=str(output_dir),
                log_model=True,
            )  # type: ignore
        )
    callbacks: List[Any] = []
    if args.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                args.checkpoint_dir,
                save_last=True,
                save_top_k=args.checkpoint_keep_count,
                every_n_epochs=1,
                monitor="metrics/test-top1-accuracy",
                mode="max",
                filename="{epoch:03d}",
            )
        )

    # providing our own progress bar disables the default progress bar (not needed to disable later on)
    cmd_logger = CommandLineLogger(args.log_interval)
    callbacks.append(cmd_logger)
    configure_logging(cmd_logger.logger, args.log_file, args.log_level, args.log_stdout)

    # add scaling callback for progressive sign (not be needed for all models, but should not slow down training)
    callbacks.append(ProgressiveSignScalerCallback())

    if len(loggers) > 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    dataset = dataset_from_name(args.dataset)

    model_kwargs = vars(model_args)
    logger.debug(f"got model args as dict: {model_kwargs}")

    model = model_from_name(args.model)(**model_kwargs, dataset=dataset)  # type: ignore
    model.initialize()

    wrapper_class: Type[ModelWrapper] = ModelWrapper
    if args.teacher:
        if args.dataset != "imagenet":
            raise ValueError(
                f"Teacher '{args.teacher}' and dataset '{args.dataset}' selected, "
                f"but teacher models are only supported for imagenet."
            )
        wrapper_class = DistillationModelWrapper

    if args.checkpoint_load is not None and args.pretrained:
        logger.info(f"starting training from pretrained model at checkpoint {args.checkpoint_load}")
        model_wrapped = wrapper_class.load_from_checkpoint(args.checkpoint_load)
    else:
        model_wrapped = wrapper_class(model, dataset.num_classes, args)

    trainer = Trainer(
        strategy=args.strategy,
        accelerator="cpu" if args.cpu else args.accelerator,
        gpus=0 if args.cpu else args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=loggers if len(loggers) > 0 else None,  # type: ignore
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=args.log_interval,
        limit_train_batches=0.01 if args.dev_run else None,
        limit_val_batches=0.01 if args.dev_run else None,
    )
    augmentation_level = Augmentation.from_string(args.augmentation)
    if args.dev_run:
        logger.info("This run only uses 1 % of training and validation data (--dev-run)!")
    logger.info(f"model: {args.model}")
    logger.info(f"optimizer: {args.optimizer}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"max_epochs: {args.max_epochs}")
    if args.fake_data:
        logger.info(f"dummy dataset: {dataset.name} (not using real data!)")
        train_dataset, test_dataset = dataset.get_dummy_train_and_test_datasets()  # type: ignore
    else:
        logger.info(f"dataset: {dataset.name}")
        train_dataset, test_dataset = dataset.get_train_and_test(  # type: ignore
            root_directory=args.dataset_dir, download=args.download, augmentation=augmentation_level
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )  # type: ignore
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    )  # type: ignore

    data_point = torch.zeros(dataset.shape)
    computational_intensity = fv_nn.FlopCountAnalysis(model, inputs=data_point, quantization_base_class=Quantization)

    stats, table = fv_nn.flop_count_table(computational_intensity, automatic_qmodules=True)
    logger.info("\n" + table)
    total_size = stats["#compressed size in bits"][""]
    logger.info("Total size in MB: " + str(total_size / 1e6 / 8.0))
    total_flops = stats["#speed up flops (app.)"][""]
    logger.info("Approximated mflops: " + str(total_flops / 1e6))
    if args.wandb_log:
        wandb.config.update(
            {
                "mflops": total_flops / 1e6,
                "size in MB": total_size / 1e6 / 8.0,
            }
        )

    trainer.fit(
        model_wrapped,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=args.checkpoint_load if not args.pretrained else None,
    )


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args_, unparsed_model_args = parser.parse_known_args()
    model_args_ = model_parser.parse_args(unparsed_model_args)

    main(args_, model_args_)
