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
import sys
import logging
from torch.utils.data import DataLoader

from utils.utils import (
    create_optimizer,
    create_scheduler,
    set_logging,
)
from utils.result_logger import ResultLogger
from utils.checkpoint_manager import CheckpointManager
from utils.experiment_creator import ExperimentCreator
from utils.eta_estimator import ETAEstimator
from utils.arg_parser import create_argparser
from train import train_model

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

    result_logger = ResultLogger(args.result_file, args.tensorboard, args.tensorboard_output)
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, args.checkpoint_keep_count)
    eta_estimator = ETAEstimator(args.eta_file, args.log_interval)

    dataset = dataset_from_name(args.dataset)
    if args.fake_data:
        logging.info(f"dummy dataset: {dataset.name} (not using real data!)...")
        train_loader, test_loader = dataset.get_dummy_train_and_test_loaders(args.batch_size)
    elif dataset.name == 'imagenet' and args.nv_dali:
        raise RuntimeError("DALI Loader not currently supported.")
    else:
        augmentation_level = Augmentation.from_string(args.augmentation)
        logging.info(f"dataset: {dataset.name}...")
        train_dataset, test_dataset = dataset.get_train_and_test(
            root_directory=args.dataset_dir, download=args.download, augmentation=augmentation_level
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True)  # type: ignore
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True)  # type: ignore

    model_kwargs = vars(model_args)
    logging.debug(f"got model args as dict: {model_kwargs}")

    model_name = args.model.lower()
    model = model_from_name(model_name)(**model_kwargs, dataset=dataset)  # type: ignore
    model.initialize()
    logging.info(f"using {model.name} model...")

    optimizer = create_optimizer(args.optimizer, model, args.lr, args.momentum)
    scheduler = create_scheduler(args.lr_scheduler, optimizer, args.lr_factor,
                                 args.lr_steps, args.epochs)  # type: ignore

    if args.checkpoint_load:
        model, optimizer, scheduler, start_epoch = checkpoint_manager.load_checkpoint(
            args.checkpoint_load, model, optimizer, scheduler, args.pretrained)
    else:
        start_epoch = 0
    gpus = False if args.cpu or args.gpus is None else args.gpus
    train_model(model, train_loader, test_loader, start_epoch=start_epoch, epochs=args.epochs, optimizer=optimizer,
                scheduler=scheduler, lr=args.lr, log_interval=args.log_interval, gpus=gpus,
                result_logger=result_logger, checkpoint_manager=checkpoint_manager, eta_estimator=eta_estimator)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    if args.experiment:
        experiment_creator = ExperimentCreator(args.experiment_name, args.experiment_dir, __file__)
        experiment_creator.create(parser, args, model_parser, model_args)
        experiment_creator.run_experiment_in_subprocess()
        sys.exit(0)

    main(args, model_args)
