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
from typing import List, Any

import fvbitcore.nn as fv_nn
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, LightningLoggerBase, WandbLogger
from torch.utils.data import DataLoader

from bitorch import apply_args_to_configuration
from bitorch.datasets import dataset_from_name
from bitorch.models import DLRM
from bitorch.quantizations import Quantization

from utils.arg_parser import create_argparser
from utils.lightning_model import ModelWrapper
from utils.utils import configure_logging
from utils.log import CommandLineLogger

from criteo import Criteo, SplitCriteoDataset
from facebook_dataloading.dataloading_fb import collate_wrapper_criteo_offset
from bitorch.datasets import datasets_by_name

logger = logging.getLogger()


def make_dlrm_dataloaders(dataset_dir, download, ignore_size, batch_size, batch_size_test, num_workers):
    # train_dataset, test_dataset = Criteo.get_train_and_test(  # type: ignore
    #     root_directory=dataset_dir, download=download, augmentation=None
    # )
    logging.info("loading Criteo dataset...")
    dataset = Criteo(True, root_directory=dataset_dir, download=download, augmentation=None).dataset

    # train_indices = np.arange(len(train_dataset))
    # np.random.shuffle(train_indices)
    # train_subset_random_sampler = SubsetRandomSampler(train_indices[:int((1 - ignore_size) * len(train_dataset))])

    # val_indices = np.arange(len(test_dataset))
    # np.random.shuffle(val_indices)
    # val_subset_random_sampler = SubsetRandomSampler(val_indices[:int((1 - ignore_size) * len(test_dataset))])

    train_dataset = SplitCriteoDataset(dataset, "train", ignore_size=ignore_size)
    test_dataset = SplitCriteoDataset(dataset, "test", ignore_size=ignore_size)
    logging.info(f"loaded {len(train_dataset)} train and {len(test_dataset)} test samples!")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wrapper_criteo_offset,
        pin_memory=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_wrapper_criteo_offset,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader, test_loader, train_dataset.dataset.m_den, train_dataset.dataset.counts


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset.

    Args:
        args (argparse.Namespace): cli arguments
        model_args (argparse.Namespace): model specific cli arguments
    """
    configure_logging(logger, args.log_file, args.log_level, args.log_stdout)

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
            WandbLogger(
                project=args.wandb_project,
                log_model=True,
                name=args.wandb_experiment,
                save_dir=str(output_dir),
            )  # type: ignore
        )
    callbacks: List[Any] = []
    if args.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                args.checkpoint_dir,
                save_last=True,
                save_top_k=args.checkpoint_keep_count,
                monitor="metrics/roc-auc",
            )
        )

    # providing our own progress bar disables the default progress bar (not needed to disable later on)
    cmd_logger = CommandLineLogger(args.log_interval)
    callbacks.append(cmd_logger)
    configure_logging(cmd_logger.logger, args.log_file, args.log_level, args.log_stdout)

    if len(loggers) > 0:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    datasets_by_name["criteo"] = Criteo
    dataset = dataset_from_name(args.dataset)
    if args.dataset == "criteo":
        train_loader, test_loader, dense_feature_size, embedding_layer_sizes = make_dlrm_dataloaders(
            args.dataset_dir,
            args.download,
            args.ignore_dataset_size,
            args.batch_size,
            args.batch_size_test,
            args.num_workers,
        )
    else:
        logging.error(f"dataset {args.dataset} is not yet supported for dlrm")
        return

    model_kwargs = vars(model_args)
    logger.debug(f"got model args as dict: {model_kwargs}")

    model = DLRM(**model_kwargs, embedding_layer_sizes=embedding_layer_sizes, dataset=dataset, dense_feature_size=dense_feature_size)  # type: ignore
    model.initialize()
    if args.checkpoint_load is not None and args.pretrained:
        logger.info(f"starting training from pretrained model at checkpoint {args.checkpoint_load}")
        model_wrapped = ModelWrapper.load_from_checkpoint(args.checkpoint_load)
    else:
        model_wrapped = ModelWrapper(model, 1, args)

    trainer = Trainer(
        strategy=args.strategy,
        accelerator="cpu" if args.cpu else args.accelerator,
        gpus=0 if args.cpu else args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=loggers if len(loggers) > 0 else None,  # type: ignore
        callbacks=callbacks,  # type: ignore
        log_every_n_steps=args.log_interval,
    )
    # augmentation_level = Augmentation.from_string(args.augmentation)
    logger.info(f"model: {args.model}")
    logger.info(f"optimizer: {args.optimizer}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"max_epochs: {args.max_epochs}")
    # if args.fake_data:
    #     logger.info(f"dummy dataset: {dataset.name} (not using real data!)")
    #     train_dataset, test_dataset = dataset.get_dummy_train_and_test_datasets()  # type: ignore
    # else:
    #     logger.info(f"dataset: {dataset.name}")
    #     train_dataset, test_dataset = dataset.get_train_and_test(  # type: ignore
    #         root_directory=args.dataset_dir, download=args.download, augmentation=augmentation_level
    #     )
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=True,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )  # type: ignore
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     shuffle=False,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )  # type: ignore

    data_point = iter(train_loader).next()
    data_point = (data_point[0], (data_point[1], data_point[2]))
    # data_point = torch.zeros(iter(train_loader).next().shape)
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
