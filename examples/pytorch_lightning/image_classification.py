import os

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
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from utils.utils import set_logging
from utils.arg_parser import create_argparser
from utils.lightning_model import ModelWrapper

from bitorch.datasets.base import Augmentation
from bitorch.models import model_from_name
from bitorch.datasets import dataset_from_name
from bitorch import apply_args_to_configuration


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset.

    Args:
        args (argparse.Namespace): cli arguments
        model_args (argparse.Namespace): model specific cli arguments
    """
    set_logging(args.log_file, args.log_level, args.log_stdout)

    apply_args_to_configuration(args)
    logging.info(f"gpu argument: {args.gpus}")
    if args.gpus is not None and len(args.gpus) == 0:
        logging.info("no specific gpu specified! Using all available gpus...")
        args.gpus = [str(i) for i in list(range(torch.cuda.device_count()))]
        logging.info(f"Using gpus: {args.gpus}")

    loggers = []
    if args.tensorboard:
        loggers.append(TensorBoardLogger(args.tensorboard_output))
    if args.result_file is not None:
        loggers.append(CSVLogger(args.result_file))
    callbacks = []
    if args.checkpoint_dir is not None:
        callbacks.append(ModelCheckpoint(args.checkpoint_dir, save_last=True,
                         save_top_k=args.checkpoint_keep_count, monitor="metrics/top1 accuracy"))

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
            model, args.optimizer, args.lr, args.momentum, args.lr_scheduler, args.lr_factor, args.lr_steps, dataset.num_classes, args.max_epochs,
        )

    trainer = Trainer(
        strategy=args.strategy,
        accelerator="cpu" if args.cpu else args.accelerator,
        gpus=0 if args.cpu else args.gpus,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        logger=loggers if len(loggers) > 0 else None,
        callbacks=callbacks,
        log_every_n_steps=args.log_interval,
    )
    # if args.checkpoint_load:
    #     model, optimizer, scheduler, start_epoch = checkpoint_manager.load_checkpoint(
    #         args.checkpoint_load, model, optimizer, scheduler, args.pretrained)
    # else:
    #     start_epoch = 0

    # if args.nv_dali and (not args.distributed_mode == "ddp" or len(args.gpus) <= 1):
    #     if args.dataset != "imagenet":
    #         raise ValueError("dali preprocessing is currently only supported for the imagenet dataset")
    #     train_dataset, test_dataset = dataset.get_train_and_test(
    #         root_directory=args.dataset_dir, download=args.download
    #     )
    #     logging.info(f"dataset: {dataset.name} (with DALI data loader)...")
    #     train_loader, test_loader = create_dali_data_loader(
    #         train_dataset.get_data_dir(), test_dataset.get_data_dir(), args.nv_dali_gpu_id, 1,
    #         args.nv_dali_cpu, args.batch_size, args.num_workers,
    #     )
    # else:
    augmentation_level = Augmentation.from_string(args.augmentation)
    if args.fake_data:
        logging.info(f"dummy dataset: {dataset.name} (not using real data!)...")
        train_dataset, test_dataset = dataset.get_dummy_train_and_test_datasets()
    else:
        logging.info(f"dataset: {dataset.name}...")
        train_dataset, test_dataset = dataset.get_train_and_test(
            root_directory=args.dataset_dir, download=args.download, augmentation=augmentation_level
        )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              shuffle=True, pin_memory=True)  # type: ignore
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=True)  # type: ignore
    trainer.fit(
        model_wrapped,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader,
        ckpt_path=args.checkpoint_load if not args.pretrained else None
    )

    # if args.distributed_mode == "ddp" and (args.world_size > 1 or (args.gpus is not None and len(args.gpus) > 1)):
    #     logging.info("Starting distributed model training...")
    #     if args.world_size < len(args.gpus):
    #         logging.warning("Total number of processes to spawn across nodes(world size) is smaller than number of"
    #                         f"gpus. Setting world size to {len(args.gpus)}")
    #         args.world_size = len(args.gpus)
    #     set_distributed_default_values(args.supervisor_host, args.supervisor_port)
    #     multiprocessing.spawn(train_model_distributed, nprocs=args.world_size,
    #                           args=(
    #                               model, train_dataset, test_dataset, result_logger, checkpoint_manager,
    #                               eta_estimator, optimizer, scheduler, args.gpus, args.base_rank, args.world_size,
    #                               start_epoch, args.epochs, args.lr, args.log_interval, args.log_file,
    #                               args.log_level, args.log_stdout, args.nv_dali, args.nv_dali_cpu, args.batch_size, args.num_workers))
    #     logging.info("Training completed!")
    # else:
    #     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                               shuffle=True, pin_memory=True)  # type: ignore
    #     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                              shuffle=False, pin_memory=True)  # type: ignore
    #     if args.distributed_mode == "dp" and len(args.gpus) > 1:
    #         logging.info("Using DataParallel multi gpu strategy...")
    #         model._model = DataParallel(model._model, device_ids=[f"cuda:{gpu_id}" for gpu_id in args.gpus])
    #     gpu = None if args.cpu or args.gpus is None else args.gpus[0]
    # train_model(model, train_loader, test_loader, start_epoch=start_epoch, epochs=args.epochs, optimizer=optimizer,
    #             scheduler=scheduler, lr=args.lr, log_interval=args.log_interval, gpu=gpu,
    #             result_logger=result_logger, checkpoint_manager=checkpoint_manager, eta_estimator=eta_estimator)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    main(args, model_args)
