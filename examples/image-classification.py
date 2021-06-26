import argparse
import sys
import logging


sys.path.append("../")

from arg_parser import create_argparser
from train import train_model
from bitorch.datasets.base import Augmentation
from bitorch.models import model_from_name
from torch.utils.data import DataLoader
from bitorch.datasets import dataset_from_name


def set_logging(args):
    log_level_name = args.log_level.upper()
    log_level = getattr(logging, log_level_name)

    if args.log_file is None:
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=log_level, force=True)
    else:
        logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, force=True)


def main(args: argparse.Namespace, model_args: argparse.Namespace) -> None:
    set_logging(args)

    dataset = dataset_from_name(args.dataset)
    augmentation_level = Augmentation.from_string(args.augmentation)
    logging.info(f"using {dataset.name} dataset...")
    train_dataset = dataset(train=True, directory=args.dataset_train_dir,
                            download=args.download, augmentation=augmentation_level)
    test_dataset = dataset(train=False, directory=args.dataset_test_dir, download=args.download)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    model_arg_dict = vars(model_args)
    logging.info(f"got model args as dict: {model_arg_dict}")
    model = model_from_name(args.model)(**model_arg_dict, dataset=dataset)
    logging.info(f"using {model.name} model...")
    gpus = False if args.cpu or not args.gpus else ','.join(args.gpus)
    train_model(model, train_loader, test_loader, epochs=args.epochs, optimizer_name=args.optimizer,
                lr_scheduler=args.lr_scheduler, lr_factor=args.lr_factor, lr_steps=args.lr_steps,
                momentum=args.momentum,
                lr=args.lr, log_interval=args.log_interval, gpus=gpus)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args, unparsed_model_args = parser.parse_known_args()
    model_args = model_parser.parse_args(unparsed_model_args)

    main(args, model_args)
