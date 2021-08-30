import argparse
import sys
import logging
from torch.utils.data import DataLoader

from utils.utils import (
    create_optimizer,
    create_scheduler,
    set_logging,
)
from utils.resultlogger import ResultLogger
from utils.checkpointmanager import CheckpointManager
from utils.experimentcreator import ExperimentCreator
from utils.etaestimator import ETAEstimator
from utils.arg_parser import create_argparser
from train import train_model


sys.path.append("../../")

from bitorch.datasets.base import Augmentation  # noqa: E402
from bitorch.models import model_from_name  # noqa: E402
from bitorch.datasets import dataset_from_name  # noqa: E402


def main(
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        model_parser: argparse.ArgumentParser,
        model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset.

    Args:
        parser (argparse.ArgumentParser): parser for cli args (used for experiment creation and)
        args (argparse.Namespace): cli arguments
        model_parser (argparse.ArgumentParser): parser for model cli args (used for experiment creation)
        model_args (argparse.Namespace): model specific cli arguments
    """
    set_logging(args.log_file, args.log_level, args.log_stdout)
    if args.experiment:
        experimentCreator = ExperimentCreator(args.experiment_name, args.experiment_dir, __file__)
        experimentCreator.create(parser, args, model_parser, model_args)
        experimentCreator.run_experiment()

    result_logger = ResultLogger(args.result_file, args.tensorboard, args.tensorboard_output)
    checkpoint_manager = CheckpointManager(args.checkpoint_dir, args.checkpoint_keep_count)
    eta_estimator = ETAEstimator(args.eta_file, args.log_interval)

    dataset = dataset_from_name(args.dataset)
    if dataset.name == 'imagenet' and args.nv_dali:
        from examples.image_classification.dali_helper import create_dali_data_loader

        logging.info(f"using {dataset.name} dataset and NV-DALI data loader ...")
        train_loader, test_loader = create_dali_data_loader(args)
    else:
        augmentation_level = Augmentation.from_string(args.augmentation)
        logging.info(f"using {dataset.name} dataset...")
        train_dataset = dataset(train=True, directory=args.dataset_train_dir,
                                download=args.download, augmentation=augmentation_level)  # type: ignore
        test_dataset = dataset(train=False, directory=args.dataset_test_dir, download=args.download)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model_arg_dict = vars(model_args)
    logging.debug(f"got model args as dict: {model_arg_dict}")

    model = model_from_name(args.model)(**model_arg_dict, dataset=dataset)  # type: ignore
    logging.info(f"using {model.name} model...")

    optimizer = create_optimizer(args.optimizer, model, args.lr, args.momentum)
    scheduler = create_scheduler(args.lr_scheduler, optimizer, args.lr_factor,
                                 args.lr_steps, args.epochs)  # type: ignore

    model, optimizer, scheduler, start_epoch = checkpoint_manager.load_checkpoint(
        args.checkpoint_load, model, optimizer, scheduler, args.fresh_start)

    gpus = False if args.cpu or not args.gpus else ','.join(args.gpus)
    train_model(model, train_loader, test_loader, start_epochs=start_epoch, epochs=args.epochs, optimizer=optimizer,
                scheduler=scheduler, lr=args.lr, log_interval=args.log_interval, gpus=gpus,
                result_logger=result_logger, checkpoint_manager=checkpoint_manager, eta_estimator=eta_estimator)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    main(parser, args, model_parser, model_args)
