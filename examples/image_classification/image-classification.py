import argparse
import sys
import logging
from utils import set_logging, ResultLogger, CheckpointManager, ExperimentCreator, ETAEstimator


sys.path.append("../../")

from arg_parser import create_argparser  # noqa: E402
from train import train_model  # noqa: E402
from bitorch.datasets.base import Augmentation  # noqa: E402
from bitorch.models import model_from_name  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from bitorch.datasets import dataset_from_name  # noqa: E402


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    """trains a model on the configured image dataset

    Args:
        args (argparse.Namespace): cli arguments
        model_args (argparse.Namespace): cli model specific arguments
    """
    set_logging(args)
    if args.experiment:
        experimentCreator = ExperimentCreator(args.experiment_name, args.experiment_dir)
        experimentCreator.create()
        experimentCreator.run_experiment()

    resultLogger = ResultLogger(args.result_file, args.tensorboard, args.tensorboard_output)
    checkpointManager = CheckpointManager(args.checkpoint_store_dir, args.checkpoint_keep_count)
    etaEstimator = ETAEstimator(args.eta_file, args.log_interval)

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
    logging.info(f"got model args as dict: {model_arg_dict}")

    model = model_from_name(args.model)(**model_arg_dict, dataset=dataset)  # type: ignore
    logging.info(f"using {model.name} model...")

    gpus = False if args.cpu or not args.gpus else ','.join(args.gpus)
    train_model(model, train_loader, test_loader, epochs=args.epochs, optimizer_name=args.optimizer,
                lr_scheduler=args.lr_scheduler, lr_factor=args.lr_factor, lr_steps=args.lr_steps,
                momentum=args.momentum,
                lr=args.lr, log_interval=args.log_interval, gpus=gpus,
                resultLogger=resultLogger, checkpointManager=checkpointManager, etaEstimator=etaEstimator)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    main(args, model_args)
