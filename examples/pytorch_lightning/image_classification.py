import os
from pathlib import Path

import torch

from examples.pytorch_lightning.utils.log import LoggingProgressBar

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    pydevd_pycharm.settrace(
        'localhost',
        port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT', "12345")),
        stdoutToServer=True,
        stderrToServer=True
    )

import argparse
import logging
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from utils.utils import set_logging
from utils.arg_parser import create_argparser
from utils.lightning_model import ModelWrapper

from bitorch.datasets.base import Augmentation
from bitorch.models import model_from_name
from bitorch.datasets import dataset_from_name
from bitorch import apply_args_to_configuration
from bitorch.quantizations import Quantization

FVBITCORE_AVAILABLE = True
try:
    import fvbitcore.nn as fv_nn
except ModuleNotFoundError:
    logging.warning("fvbitcore not installed, will not calculate model flops!")
    FVBITCORE_AVAILABLE = False

WANDB_AVAILABLE = True
try:
    from pytorch_lightning.loggers import WandbLogger
    import wandb
except ModuleNotFoundError:
    logging.warning("wandb not installed, will not log metrics to wandb!")
    WANDB_AVAILABLE = False


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset.

    Args:
        args (argparse.Namespace): cli arguments
        model_args (argparse.Namespace): model specific cli arguments
    """
    set_logging(args.log_file, args.log_level, args.log_stdout)

    apply_args_to_configuration(args)

    output_dir = Path(args.result_directory)
    output_dir.mkdir(exist_ok=True)

    loggers = []
    if args.tensorboard_log:
        loggers.append(TensorBoardLogger(output_dir, name="tensorboard"))  # type: ignore
    if args.csv_log:
        loggers.append(CSVLogger(output_dir, name="csv"))  # type: ignore
    if WANDB_AVAILABLE and args.wandb_log:
        try:
            loggers.append(
                WandbLogger(project=args.wandb_project, log_model=True, name=args.wandb_experiment, save_dir=str(output_dir))
            )  # type: ignore
        except ModuleNotFoundError:
            logging.warning(
                "wandb is not installed, values will not be logged via wandb. install it with "
                "`pip install wandb`."
            )
    callbacks = []
    if args.checkpoint_dir is not None:
        callbacks.append(ModelCheckpoint(args.checkpoint_dir, save_last=True,
                         save_top_k=args.checkpoint_keep_count, monitor="metrics/test-top1-accuracy"))

    # providing our own progress bar disables the default progress bar (not needed to disable later on)
    callbacks.append(LoggingProgressBar(args.log_interval))

    if len(loggers) > 0:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    dataset = dataset_from_name(args.dataset)

    model_kwargs = vars(model_args)
    logging.debug(f"got model args as dict: {model_kwargs}")

    model = model_from_name(args.model)(**model_kwargs, dataset=dataset)  # type: ignore
    model.initialize()
    if args.checkpoint_load is not None and args.pretrained:
        logging.info(f"starting training from pretrained model at checkpoint {args.checkpoint_load}")
        model_wrapped = ModelWrapper.load_from_checkpoint(args.checkpoint_load)
    else:
        model_wrapped = ModelWrapper(
            model, args.optimizer, args.lr, args.momentum, args.lr_scheduler, args.lr_factor, args.lr_steps,
            dataset.num_classes, args.max_epochs,
        )

    trainer = Trainer(
        strategy=args.strategy,
        accelerator="cpu" if args.cpu else args.accelerator,
        gpus=0 if args.cpu else args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=loggers if len(loggers) > 0 else None,  # type: ignore
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=args.log_interval
    )
    augmentation_level = Augmentation.from_string(args.augmentation)
    logging.info(f"model: {args.model}")
    logging.info(f"optimizer: {args.optimizer}")
    logging.info(f"lr: {args.lr}")
    logging.info(f"max_epochs: {args.max_epochs}")
    if args.fake_data:
        logging.info(f"dummy dataset: {dataset.name} (not using real data!)")
        train_dataset, test_dataset = dataset.get_dummy_train_and_test_datasets()  # type: ignore
    else:
        logging.info(f"dataset: {dataset.name}")
        train_dataset, test_dataset = dataset.get_train_and_test(  # type: ignore
            root_directory=args.dataset_dir, download=args.download, augmentation=augmentation_level
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True, persistent_workers=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True, persistent_workers=True)  # type: ignore

    if FVBITCORE_AVAILABLE:
        data_point = torch.zeros(dataset.shape)
        computational_intensity = fv_nn.FlopCountAnalysis(
            model,
            inputs=data_point,
            quantization_base_class=Quantization
        )

        stats, table = fv_nn.flop_count_table(computational_intensity, automatic_qmodules=True)
        logging.info("\n" + table)
        total_size = stats["#compressed size in bits"][""]
        logging.info("Total size in MB: " + str(total_size / 1e6 / 8.0))
        total_flops = stats["#speed up flops (app.)"][""]
        logging.info("Approximated mflops: " + str(total_flops / 1e6))
        # for logger in loggers:
        #     logger.log_dict({
        #         "mflops": total_flops / 1e6,
        #         "size in MB": total_size / 1e6 / 8.0,
        #     })

    trainer.fit(
        model_wrapped,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=args.checkpoint_load if not args.pretrained else None
    )


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    main(args, model_args)
