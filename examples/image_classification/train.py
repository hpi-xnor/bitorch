import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
import logging
from typing import Union, Optional
from math import floor
from bitorch.optimization.radam import RAdam


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
    if name == "adam":
        return Adam(params=model.parameters(), lr=lr)
    elif name == "sgd":
        return SGD(params=model.parameters(), lr=lr, momentum=momentum)
    elif name == "radam":
        return RAdam(params=model.parameters(), lr=lr, degenerated_to_sgd = False)
    else:
        raise ValueError(f"No optimizer with name {name} found!")


def create_scheduler(
        scheduler_name: Optional[str],
        optimizer: Optimizer,
        lr_factor: float,
        lr_steps: Optional[list],
        epochs: int) -> Union[_LRScheduler, None]:
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


def train_model(
        model: Module,
        train_data: DataLoader,
        test_data: DataLoader,
        epochs: int = 10,
        optimizer_name: str = "adam",
        lr_scheduler: str = None,
        lr_factor: float = 0.1,
        lr_steps: str = None,
        momentum: float = 0.9,
        lr: float = 0.001,
        log_interval: int = 100,
        gpus: str = None) -> Module:
    """trains the given model on the given train and test data. creates optimizer and lr scheduler with the given params.
    in each epoch validation on the test data is performed. gpu acceleration can be enabled.

    Args:
        model (Module): the Model to be trained
        train_data (DataLoader): train dataloader
        test_data (DataLoader): test dataloader
        epochs (int, optional): number of epochs to train the model. Defaults to 10.
        optimizer_name (str, optional): str name of optimizer. Defaults to "adam".
        lr_scheduler (str, optional): str name of lr scheduler. if None, no lr scheduling will be performed.
            Defaults to None.
        lr_factor (float, optional): lr factor to be applied by lr scheduler. Defaults to 0.1.
        lr_steps (str, optional): lr steps for the lr scheduler to take. only supported with the step lr scheduler.
            Defaults to None.
        momentum (float, optional): momentum value to be used by sgd optimizer. Defaults to 0.9.
        lr (float, optional): learning rate to be used by optimizer. Defaults to 0.001.
        log_interval (int, optional): interval at wich logging info shall be outputed. Defaults to 100.
        gpus (str, optional): list of gpus to be used during training. if None, training will be performed on cpu.
            Defaults to None.

    Returns:
        Module: the trained model
    """

    criterion = CrossEntropyLoss()
    if gpus:
        device = "cuda:" + gpus
    else:
        device = "cpu"
    model = model.to(device)

    optimizer = create_optimizer(optimizer_name, model, lr, momentum)
    scheduler = create_scheduler(lr_scheduler, optimizer, lr_factor, lr_steps, epochs)  # type: ignore

    total_number_of_batches = epochs * len(train_data)
    current_number_of_batches = 0

    for epoch in range(epochs):
        epoch_loss = 0.0

        model.train()
        for idx, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_hat = model(x_train)
            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            current_number_of_batches += 1

            if idx % log_interval == 0 and idx > 0:
                progress = floor((current_number_of_batches / total_number_of_batches) * 100000.0) / 1000.0
                logging.info(
                    f"    (Progress: {progress}%) Loss in epoch {epoch + 1} for batch {idx}: {epoch_loss / idx}, "
                    f"current lr: {scheduler.get_last_lr() if scheduler else lr}")
        epoch_loss /= len(train_data)

        if scheduler:
            scheduler.step()

        model.eval()
        test_loss = 0.0
        correct = 0.0
        # now validate model with test dataset
        with torch.no_grad():
            for idx, (x_test, y_test) in enumerate(test_data):
                x_test = x_test.to(device)
                y_test = y_test.to(device)

                y_hat = model(x_test)
                test_loss += criterion(y_hat, y_test).item()

                # determine count of correctly predicted labels
                predictions = torch.argmax(y_hat, dim=1)
                correct += torch.sum(y_test == predictions).item()
        test_loss /= len(test_data)
        batch_size = test_data.batch_size
        if batch_size is None:
            batch_size = 1
        accuracy = correct / (len(test_data) * batch_size)

        logging.info(
            f"Epoch {epoch + 1} train loss: {epoch_loss}, test loss: {test_loss}, "
            f"test accuracy: {accuracy}")

    return model
