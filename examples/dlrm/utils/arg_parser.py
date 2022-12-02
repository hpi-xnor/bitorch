from argparse import ArgumentParser
import sys
from typing import Tuple

from bitorch.models import model_from_name, model_names
from bitorch import add_config_args
from bitorch.models.dlrm import DLRM
from pytorch_lightning import Trainer


def add_logging_args(parser: ArgumentParser) -> None:
    """adds cli parameters for logging configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    log = parser.add_argument_group("Logging", "parameters for logging")
    log.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="log level for logging message output",
    )
    log.add_argument(
        "--log-interval",
        type=int,
        default=100,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    log.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="output file path for logging. default to stdout",
    )
    log.add_argument(
        "--log-stdout",
        action="store_true",
        help="toggles force logging to stdout. if a log file is specified, logging will be"
        "printed to both the log file and stdout",
    )
    log.add_argument(
        "--result-directory",
        type=str,
        default="./logs",
        help="path to logs directory, e.g. tensorboard logs, csv files",
    )
    log.add_argument(
        "--disable-tensorboard-log",
        action="store_false",
        dest="tensorboard_log",
        help="disables tensorboard logging",
    )
    log.add_argument(
        "--disable-csv-log",
        action="store_false",
        dest="csv_log",
        help="disables csv logging",
    )
    log.add_argument(
        "--wandb",
        action="store_true",
        dest="wandb_log",
        help="enables wandb logging (WANDB_API_KEY environment variable must be set)",
    )
    log.add_argument(
        "--wandb-project",
        type=str,
        default="bitorch",
        help="name of wand project to be used by wandb logger",
    )
    log.add_argument(
        "--wandb-experiment",
        type=str,
        default=None,
        help="name of wand experiment to be used by wandb logger",
    )


def add_checkpoint_args(parser: ArgumentParser) -> None:
    """adds cli parameters for checkpoint logging

    Args:
        parser (ArgumentParser): the main argument parser
    """
    checkpoint = parser.add_argument_group("checkpoints", "parameters for checkpoint storing / loading")
    checkpoint.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="set a custom path to store checkpoints in.",
    )
    checkpoint.add_argument(
        "--checkpoint-keep-count",
        type=int,
        default=1,
        help="number of checkpoints to keep.",
    )
    checkpoint.add_argument(
        "--checkpoint-load",
        type=str,
        default=None,
        help="path to checkpoint file to load state from. if omitted and --pretrained is activated, the model weights will be downloaded from the model hub. If --pretrained is not set, a new model will be trained.",
    )
    checkpoint.add_argument(
        "--resume_training",
        action="store_true",
        help="resume training from given checkpoint",
    )
    checkpoint.add_argument(
        "--pretrained",
        action="store_true",
        help="load the state dict either from model hub or from checkpoint_load",
    )


def add_optimizer_args(parser: ArgumentParser) -> None:
    """adds cli parameters for optimizer configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    optimizer = parser.add_argument_group("Optimizer", "parameters for optimizer")
    optimizer.add_argument(
        "--lr-scheduler",
        type=str,
        choices=["cosine", "step", "exponential"],
        help="name of the lr scheduler to use. default to none",
    )
    optimizer.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="initial learning rate (default: 0.01)",
    )
    optimizer.add_argument(
        "--lr-factor",
        default=0.1,
        type=float,
        help="learning rate decay ratio. this is used only by the step and exponential lr scheduler",
    )
    optimizer.add_argument(
        "--lr-steps",
        nargs="*",
        default=[30, 60, 90],
        help="list of learning rate decay epochs as list. this is used only by the step scheduler",
    )
    optimizer.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum value for optimizer, default is 0.9. only used for sgd optimizer",
    )
    optimizer.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "radam"],
        help="the optimizer to use. default is adam.",
    )


def add_dataset_args(parser: ArgumentParser) -> None:
    """adds cli parameters for dataset configuration

    Args:
        parser (ArgumentParser): the main argument parser
    """
    data = parser.add_argument_group("dataset", "parameters for the dataset used for training")
    data.add_argument(
        "--dataset",
        type=str,
        default="criteo",
        choices=["criteo"],
        help="name of the dataset to be used for training",
    )
    data.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="path to where the train dataset is saved / shall be downloaded to",
    )
    data.add_argument(
        "--download",
        action="store_true",
        help="toggles wether the dataset shall be downloaded if not present. "
        "only has effect with the cifar10 and mnist dataset so far.",
    )
    data.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="batch size for training (default: 128)",
    )
    data.add_argument(
        "--batch-size-test",
        type=int,
        default=128,
        help="batch size for testing (might be higher than training) (default: 128)",
    )
    data.add_argument(
        "--ignore-dataset-size",
        type=float,
        default=0.9,
        help="portion of dataset that should be ignored for training (usefull for fast development) (default: 128)",
    )
    data.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of workers to be used for dataloading (default: 4)",
    )
    data.add_argument(
        "--augmentation",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="none",
        help="level of augmentation to be used in data preparation (default 'none')",
    )
    data.add_argument(
        "--fake-data",
        action="store_true",
        help="train with fake data",
    )


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


def add_regular_args(parser: ArgumentParser) -> None:
    """adds all regular arguments, including dynamically created config args to parser.

    Args:
        parser (ArgumentParser): parser to add the regular arguments to
    """
    Trainer.add_argparse_args(parser)
    add_logging_args(parser)
    add_dataset_args(parser)
    add_optimizer_args(parser)
    add_checkpoint_args(parser)

    add_config_args(parser)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="explicitly use the cpu. overwrites gpu settings",
    )


def create_argparser(arguments=None) -> Tuple[ArgumentParser, ArgumentParser]:
    """creates a main argument parser for general options and a model parser for model specific options

    Returns:
        Tuple[ArgumentParser, ArgumentParser]: the main and model argument parser
    """
    parser = ArgumentParser(description="Bitorch Image Classification")

    add_regular_args(parser)

    if help_in_args():
        model_group = parser.add_argument_group("DLRM", "parameters for DLRM model")
        DLRM.add_argparse_arguments(model_group)
    args, _ = parser.parse_known_args()

    model_parser = create_model_argparser(DLRM)
    return parser, model_parser
