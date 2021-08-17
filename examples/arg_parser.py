from argparse import ArgumentParser
import sys
from typing import Tuple
from bitorch.models import model_from_name, model_names
from bitorch.datasets import dataset_names


def add_logging_args(parser: ArgumentParser) -> None:
    """adds cli parameters for logging configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    log = parser.add_argument_group("Logging", "parameters for logging")
    log.add_argument("--log-level", type=str, required=False, default="info",
                     choices=["debug", "info", "warning", "error", "critical"],
                     help="log level for logging message output")
    log.add_argument("--log-interval", type=int, default=100, metavar="N",
                     help="how many batches to wait before logging training status")
    log.add_argument("-l", "--log-file", type=str, required=False, default=None,
                     help="output file path for logging. default to stdout")


def add_optimizer_args(parser: ArgumentParser) -> None:
    """adds cli parameters for optimizer configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    optimizer = parser.add_argument_group("Optimizer", "parameters for optimizer")
    optimizer.add_argument("--lr-scheduler", type=str, required=False,
                           choices=["cosine", "step", "exponential"],
                           help="name of the lr scheduler to use. default to none")
    optimizer.add_argument("--lr", type=float, default=0.01,
                           help="initial learning rate (default: 0.01)")
    optimizer.add_argument('--lr-factor', default=0.1, type=float,
                           help='learning rate decay ratio. this is used only by the step and exponential lr scheduler')
    optimizer.add_argument('--lr-steps', nargs="*", required=False, default=[30, 60, 90],
                           help='list of learning rate decay epochs as list. this is used only by the step scheduler')
    optimizer.add_argument('--momentum', type=float, default=0.9,
                           help='momentum value for optimizer, default is 0.9. only used for sgd optimizer')
    optimizer.add_argument('--optimizer', type=str, required=False, default="adam", choices=["adam", "sgd"],
                           help='the optimizer to use. default is adam.')


def add_training_args(parser: ArgumentParser) -> None:
    """adds cli parameters for training configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    train = parser.add_argument_group("training", "parameters for training")
    train.add_argument("--epochs", type=int, default=10,
                       help="number of epochs to train (default: 10)")
    train.add_argument("--gpus", nargs="*", required=False,
                       help="list of GPUs to train on using CUDA. Parameter should be a list of gpu numbers, e.g. "
                       " --gpus 0 2 to train on gpus no. 0 and no. 2. if omitted, cpu training will be enforced")
    train.add_argument("--cpu", action="store_true", default=False, required=False,
                       help="explicitly use the cpu. overwrites gpu settings")


def add_dataset_args(parser: ArgumentParser) -> None:
    """adds cli parameters for dataset configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    data = parser.add_argument_group("dataset", "parameters for the dataset used for training")
    data.add_argument("--dataset", type=str, default="cifar10", choices=dataset_names(),
                      help="name of the dataset to be used for training")
    data.add_argument("--dataset-train-dir", type=str, default="./train", required=False,
                      help="path to where the train dataset is saved / shall be downloaded to")
    data.add_argument("--dataset-test-dir", type=str, default="./test", required=False,
                      help="path to where the test dataset is saved / shall be downloaded to")
    data.add_argument("--download", action="store_true", required=False, default=True,
                      help="toggles wether the dataset shall be downloaded if not present. "
                      "only has effect with the cifar10 and mnist dataset so far.")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="batch size for training and testing (default: 100)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of workers to be used for dataloading (default: 0)")
    parser.add_argument("--augmentation", type=str, choices=["none", "low", "medium", "high"], default="none",
                        help="level of augmentation to be used in data preparation (default 'none')")


def create_model_argparser(model_class: object) -> ArgumentParser:
    """adds model specific cli arguments from model_class object

    Args:
        model_class (object): the class-object of selected model

    Returns:
        ArgumentParser: cli argument parser
    """
    model_parser = ArgumentParser(add_help=False)
    model_class.add_argparse_arguments(model_parser)
    return model_parser


def help_in_args() -> bool:
    """determines if script was called with a --help or -h flag

    Returns:
        bool: True if help flag was set, False otherwise
    """
    passed_args = sys.argv[1:]
    if "--help" in passed_args or "-h" in passed_args:
        return True
    return False


def add_all_model_args(parser: ArgumentParser) -> None:
    """iterates through all existent models and adds their specific cli args to parser

    Args:
        parser (ArgumentParser): the main cli argument parser
    """
    for model_name in model_names():
        model_group = parser.add_argument_group(model_name, f"parameters for {model_name} model")
        model_from_name(model_name).add_argparse_arguments(model_group)  # type: ignore


def create_argparser() -> Tuple[ArgumentParser, ArgumentParser]:
    """creates a main argument parser for general options and a model parser for model specific options

    Returns:
        Tuple[ArgumentParser, ArgumentParser]: the main and model argument parser
    """
    parser = ArgumentParser(description="Bitorch Image Classification")
    add_logging_args(parser)
    add_training_args(parser)
    add_dataset_args(parser)
    add_optimizer_args(parser)

    parser.add_argument("--model", type=str, choices=model_names(),
                        help="name of the model to be trained")
    if help_in_args():
        add_all_model_args(parser)
    args, _ = parser.parse_known_args()

    model_class = model_from_name(args.model)
    model_parser = create_model_argparser(model_class)
    return parser, model_parser
