import argparse
from bitorch.datasets import dataset_from_name
from examples.arg_parser import create_argparser
import logging


def set_logging(args):
    log_level_name = args.log_level.toUpper()
    log_level = getattr(logging, log_level_name)

    if args.log_file is None:
        logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=log_level, force=True)
    else:
        logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(levelname)s: %(message)s',
                            level=logging.DEBUG, force=True)


def main(args: argparse.Namespace) -> None:
    set_logging(args)

    dataset = dataset_from_name(args.dataset)
    train_dataset = dataset(train=True, directory=args.dataset_train_dir, download=args.download)
    test_dataset = dataset(train=False, directory=args.dataset_test_dir, download=args.download)


if __name__ == "__main__":
    parser, model_parser = create_argparser()
    args = parser.parse_args()
    model_args = model_parser.parse_args

    main(args, model_args)
