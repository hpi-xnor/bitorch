import logging
from pathlib import Path

from torch.optim import Adam, SGD, RAdam
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
from typing import Union, Optional, Any
from torch.nn import Module
from torch.optim.optimizer import Optimizer


def configure_logging(logger: Any, log_file: Optional[str], log_level: str, output_stdout: bool) -> None:
    """configures logging module.

    Args:
        logger: the logger to be configured
        log_file (str): path to log file. if omitted, logging will be forced to stdout.
        log_level (str): string name of log level (e.g. 'debug')
        output_stdout (bool): toggles stdout output. will be activated automatically if no log file was given.
            otherwise if activated, logging will be outputed both to stdout and log file.
    """
    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logger.setLevel(log_level)

    logging_format = logging.Formatter(
        "%(asctime)s - %(levelname)s [%(filename)s : %(funcName)s() : l. %(lineno)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if log_file is not None:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    else:
        output_stdout = True

    if output_stdout:
        stream = logging.StreamHandler()
        stream.setFormatter(logging_format)
        logger.addHandler(stream)


def create_optimizer(name: str, model: Module, lr: float, momentum: float) -> Optimizer:
    """creates the specified optimizer with the given parameters

    Args:
        name (str): str name of optimizer
        model (Module): the model used for training
        lr (float): learning rate
        momentum (float): momentum (only for sgd optimizer)

    Raises:
        ValueError: thrown if optimizer name not known

    Returns:
        Optimizer: the model optimizer
    """
    name = name.lower()
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
    scheduler_name: Optional[str],
    optimizer: Optimizer,
    lr_factor: float,
    lr_steps: Optional[list],
    epochs: int,
) -> Union[_LRScheduler, None]:
    """creates a learning rate scheduler with the given parameters

    Args:
        scheduler_name (Optional[str]): str name of scheduler or None, in which case None will be returned
        optimizer (Optimizer): the learning optimizer
        lr_factor (float): the learning rate factor
        lr_steps (Optional[list]): learning rate steps for the scheduler to take (only supported for step scheduler)
        epochs (int): number of scheduler epochs (only supported for cosine scheduler)

    Raises:
        ValueError: thrown if step scheduler was chosen but no steps were passed
        ValueError: thrown if scheduler name not known and not None

    Returns:
        Union[_LRScheduler, None]: either the learning rate scheduler object or None if scheduler_name was None
    """
    if scheduler_name == "step":
        if not lr_steps:
            raise ValueError("step scheduler chosen but no lr steps passed!")
        return MultiStepLR(optimizer, lr_steps, lr_factor)
    elif scheduler_name == "exponential":
        return ExponentialLR(optimizer, lr_factor)
    elif scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, epochs)
    elif not scheduler_name:
        return None
    else:
        raise ValueError(f"no scheduler with name {scheduler_name} found!")
