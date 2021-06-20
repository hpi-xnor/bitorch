from argparse import ArgumentParser
import sys
from bitorch.models import model_from_name, model_names
from bitorch.datasets import dataset_names


def add_logging_args(parser: ArgumentParser) -> None:
    data = parser.add_argument_group("Logging", "parameters for logging")
    data.add_argument("--log-level", type=str, required=False, default="info",
                      choices=["debug", "info", "warning", "error", "critical"],
                      help="log level for logging message output")
    data.add_argument("--log-interval", type=int, default=100, metavar="N",
                      help="how many batches to wait before logging training status")
    data.add_argument("-l", "--log-file", type=str, required=False, default=None,
                      help="output file path for logging. default to stdout")


def add_training_args(parser: ArgumentParser) -> None:
    train = parser.add_argument_group("training", "parameters for training")
    train.add_argument("--epochs", type=int, default=10,
                       help="number of epochs to train (default: 10)")
    train.add_argument("--lr", type=float, default=0.01,
                       help="learning rate (default: 0.01)")
    train.add_argument("--gpus", nargs="*", required=False,
                       help="list of GPUs to train on using CUDA. Parameter should be a list of gpu numbers, e.g. "
                       " --gpus 0 2 to train on gpus no. 0 and no. 2. if omitted, cpu training will be enforced")
    train.add_argument("--cpu", action="store_true", default=False, required=False,
                       help="explicitly use the cpu. overwrites gpu settings")


def add_dataset_args(parser: ArgumentParser) -> None:
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
    parser.add_argument("--num-worker", type=int, default=4,
                        help="number of workers to be used for dataloading (default: 4)")
    parser.add_argument("--augmentation", type=str, choices=["none", "low", "medium", "high"], default="none",
                        help="level of augmentation to be used in data preparation (default 'none')")


def add_model_args(model_class: object):
    model_parser = ArgumentParser(help=False)
    model_class.add_argparse_arguments(model_parser)
    return model_parser


def help_in_args() -> bool:
    passed_args = sys.argv[1:]
    if "--help" in passed_args or "-h" in passed_args:
        return True
    return False


def add_all_model_args(parser: ArgumentParser):
    for model_name in model_names():
        model_group = parser.add_argument_group(model_name, f"parameters for {model_name} model")
        model_from_name(model_name).add_argparse_arguments(model_group)


def create_argparser():
    parser = ArgumentParser(description="Bitorch Image Classification")
    add_logging_args(parser)
    add_training_args(parser)
    add_dataset_args(parser)

    parser.add_argument("--model", type=str, choices=model_names(),
                        help="name of the model to be trained")
    if help_in_args():
        add_all_model_args(parser)
    args, _ = parser.parse_known_args()

    model_class = model_from_name(args.model)
    model_parser = add_model_args(model_class, parser)
    return parser, model_parser
