import argparse
import sys
from argparse import ArgumentParser
from typing import Tuple, List, Type, Any, Optional, Sequence

from pytorch_lightning import Trainer

import bitorch
from bitorch.models import model_from_name, model_names, Model
from bitorch.models.base import NoArgparseArgsMixin
from bitorch.quantizations.quantization_scheduler import Quantization_Scheduler
from datasets import dataset_names
from utils.teachers import available_teachers


class _HeadArgumentParser(ArgumentParser):
    _informational_sub_parsers: List[ArgumentParser]

    def print_help(self, *args: Any) -> None:
        super().print_help(*args)
        if hasattr(self, "_informational_sub_parsers"):
            for parser in self._informational_sub_parsers:
                print("\n")
                parser.print_help()

    def add_informational_subparsers(self, sub_parsers: List[ArgumentParser]) -> None:
        self._informational_sub_parsers = sub_parsers


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
        "--quantization-scheduling",
        action="store_true",
        default=False,
        help="toggles weather to use quantization scheduling",
    )
    optimizer.add_argument(
        "--schedule-all-quantizations",
        action="store_true",
        default=False,
        help="toggles weather to replace all quantizations inside the model with scheduled quantizers or "
        "to just use the instances of ScheduledQuantizer which are already present in the model.",
    )
    optimizer.add_argument(
        "--scheduled-quantizations",
        nargs="*",
        default=["identity", "sign"],
        help="name of quantizations to schedule",
    )
    optimizer.add_argument(
        "--quantization-scheduling-procedure",
        type=str,
        default="mix_linear",
        choices=list(Quantization_Scheduler.procedure_classes.keys()),
        help="procedure to use for scheduling",
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


def add_training_args(parser: ArgumentParser) -> None:
    """
    Add arguments for training strategies.

    Args:
        parser (ArgumentParser): the main argument parser
    """
    train = parser.add_argument_group("training", "parameters for the training strategies")
    train.add_argument(
        "--teacher",
        type=str,
        default="",
        choices=available_teachers(),
        help="name of the teacher model, the student is going to be trained with KD if not empty",
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
        default="cifar10",
        choices=dataset_names(),
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
        help="batch size for training and testing (default: 128)",
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


def create_model_argparser(model_class: Type[Model]) -> ArgumentParser:
    """adds model specific cli arguments from model_class object

    Args:
        model_class (object): the class-object of selected model

    Returns:
        ArgumentParser: cli argument parser
    """
    model_parser = argparse.ArgumentParser(
        description=f"Additional arguments for {model_class.name} (--model {model_class.name.lower()})",
        add_help=False,
        usage=argparse.SUPPRESS,
    )
    model_class.add_argparse_arguments(model_parser)
    return model_parser


def help_in_args(cmd_args: Optional[Sequence[str]] = None) -> bool:
    """determines if script was called with a --help or -h flag

    Returns:
        bool: True if help flag was set, False otherwise
    """
    passed_args = cmd_args
    if passed_args is None:
        passed_args = sys.argv[1:]
    if "--help" in passed_args or "-h" in passed_args:
        return True
    return False


def create_list_of_all_model_parsers() -> List[ArgumentParser]:
    """iterates through all existent models and adds a parser for each one"""
    all_model_parsers = []
    for model_name in model_names():
        model_class = model_from_name(model_name)
        if model_class.add_argparse_arguments == Model.add_argparse_arguments:
            continue
        if model_class.add_argparse_arguments == NoArgparseArgsMixin.add_argparse_arguments:
            continue
        model_parser = create_model_argparser(model_class)
        all_model_parsers.append(model_parser)
    return all_model_parsers


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
    add_training_args(parser)

    bitorch.add_config_args(parser)

    parser.add_argument(
        "--model",
        type=str.lower,
        choices=model_names(),
        required=True,
        help="name of the model to be trained",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="explicitly use the cpu. overwrites gpu settings",
    )
    parser.add_argument(
        "--dev-run",
        action="store_true",
        help="use only 1%% of training/validation data for testing purposes",
    )


def create_argparser(cmd_args: Optional[Sequence[str]] = None) -> Tuple[ArgumentParser, ArgumentParser]:
    """creates a main argument parser for general options and a model parser for model specific options

    Returns:
        Tuple[ArgumentParser, ArgumentParser]: the main and model argument parser
    """
    parser = _HeadArgumentParser(description="Bitorch Image Classification")

    add_regular_args(parser)

    if help_in_args(cmd_args):
        parser.add_informational_subparsers(create_list_of_all_model_parsers())
    args, _ = parser.parse_known_args(cmd_args)

    model_class = model_from_name(args.model)
    model_parser = create_model_argparser(model_class)
    return parser, model_parser
