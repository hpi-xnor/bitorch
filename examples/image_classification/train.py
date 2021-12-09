import logging
from typing import List

import torch
from torch.nn import Module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from bitorch.quantizations.base import Quantization
from utils.checkpoint_manager import CheckpointManager
from utils.eta_estimator import ETAEstimator
from utils.metrics_calculator import MetricsCalculator
from utils.result_logger import ResultLogger

try:
    from bitorchinfo import summary
except ImportError:
    summary = None  # type: ignore


def train_model(
        model: Module,
        train_data: DataLoader,
        test_data: DataLoader,
        result_logger: ResultLogger,
        checkpoint_manager: CheckpointManager,
        eta_estimator: ETAEstimator,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        start_epoch: int = 0,
        epochs: int = 10,
        lr: float = 0.001,
        log_interval: int = 100,
        gpus: List[str] = None) -> Module:
    """trains the given model on the given train and test data. creates optimizer and lr scheduler with the given params.
    in each epoch validation on the test data is performed. gpu acceleration can be enabled.

    Args:
        model (Module): the Model to be trained
        train_data (DataLoader): train dataloader
        test_data (DataLoader): test dataloader
        epochs (int, optional): number of epochs to train the model. Defaults to 10.
        lr (float, optional): learning rate to be used by optimizer. Defaults to 0.001.
        log_interval (int, optional): interval at wich logging info shall be outputed. Defaults to 100.
        gpus (str, optional): list of gpus to be used during training. if None, training will be performed on cpu.
            Defaults to None.

    Returns:
        Module: the trained model
    """

    criterion = CrossEntropyLoss()
    if gpus or isinstance(gpus, list):
        # use all available gpus if no specific gpus are listed
        if len(gpus) == 0:
            gpus = list(map(str, range(torch.cuda.device_count())))
        logging.info(f"training model on gpus: {gpus}...")
        device = "cuda:" + ",".join(gpus)
    else:
        logging.info("training model on cpu...")
        device = "cpu"
    model = model.to(device)

    # some code for model visualization / storing of initial state
    model.eval()
    images, _ = iter(train_data).next()
    images = images.to(device)
    _ = model(images)
    model.train()
    result_logger.log_model(model, images)
    checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, 0, f"{model.name}_untrained")

    if summary is None:
        logging.warning("Can not create a model summary because the package 'bitorchinfo' is not installed!")
    else:
        summary_str = summary(model, verbose=0, input_data=images, depth=10,
                              quantization_base_class=Quantization, device=device)
        logging.info(f"Model summary:\n{summary_str}")

    # initialization of eta estimator
    total_number_of_batches = (epochs - start_epoch) * (len(train_data) + len(test_data))
    eta_estimator.set_iterations(total_number_of_batches)
    current_number_of_batches = 0
    best_accuracy = 0.0

    metrics = MetricsCalculator()

    for epoch in range(start_epoch, epochs):
        eta_estimator.epoch_start()

        model.train()
        metrics.clear()
        for idx, (x_train, y_train) in enumerate(train_data):
            current_number_of_batches += 1
            with eta_estimator:
                optimizer.zero_grad()
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                y_hat = model(x_train)
                loss = criterion(y_hat, y_train)
                loss.backward()
                optimizer.step()

                metrics.update(y_hat, y_train, loss)
                result_logger.tensorboard_results(
                    category="Batch",
                    step=current_number_of_batches * len(x_train),
                    loss=metrics.avg_loss(),
                )

            if idx % log_interval == 0 and idx > 0:
                result_logger.tensorboard_results(
                    category="Batch",
                    step=current_number_of_batches * len(x_train),
                    accuracy=metrics.accuracy(),
                    recall=metrics.recall(),
                    precision=metrics.precision(),
                    f1=metrics.f1(),
                    top_5_accuracy=metrics.top_5_accuracy(),
                )
                speed_in_sample_per_s = train_data.batch_size * eta_estimator.iterations_per_second()
                lr = scheduler.get_last_lr()[0] if scheduler else lr
                logging.info(
                    f"epoch {epoch + 1:03d} batch {idx:4d}: loss: {metrics.avg_loss():.4f}, "
                    f"acc: {metrics.accuracy():.4f}, current lr: {lr:.7f}, ({speed_in_sample_per_s:.1f} samples/s, "
                    f"eta: {eta_estimator.eta()})"
                )

        result_logger.tensorboard_results(
            category="Train",
            step=epoch + 1,
            loss=metrics.avg_loss(),
            accuracy=metrics.accuracy(),
            recall=metrics.recall(),
            precision=metrics.precision(),
            f1=metrics.f1(),
            top_5_accuracy=metrics.top_5_accuracy(),
        )
        train_loss = metrics.avg_loss()

        if scheduler:
            scheduler.step()

        model.eval()
        metrics.clear()

        # now validate model with test dataset
        with torch.no_grad():
            for idx, (x_test, y_test) in enumerate(test_data):
                with eta_estimator:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)

                    y_hat = model(x_test)
                    test_loss = criterion(y_hat, y_test)
                    metrics.update(y_hat, y_test, test_loss)

        accuracy = metrics.accuracy()
        result_logger.log_result(
            epoch=epoch + 1,
            lr=scheduler.get_last_lr() if scheduler else lr,
            train_epoch_loss=train_loss,
            test_epoch_loss=metrics.avg_loss(),
            test_top1_acc=accuracy,
            test_top5_acc=metrics.top_5_accuracy(),
            test_recall=metrics.recall(),
            test_precision=metrics.precision(),
            test_f1=metrics.f1(),
        )
        result_logger.tensorboard_results(
            category="Test",
            step=epoch + 1,
            loss=metrics.avg_loss(),
            accuracy=accuracy,
            recall=metrics.recall(),
            precision=metrics.precision(),
            f1=metrics.f1(),
            top_5_accuracy=metrics.top_5_accuracy(),
        )

        # checkpoint updating
        checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, epoch)
        if accuracy > best_accuracy:
            logging.info("updating best model....")
            checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, epoch, f"{model.name}_best")
            best_accuracy = accuracy

        logging.info(eta_estimator.summary())
        eta_estimator.epoch_end()

        result_logger.tensorboard_results(
            category="Training",
            reverse_tag=True,
            step=epoch + 1,
            epoch_duration=eta_estimator.epoch_duration(),
            lr=scheduler.get_last_lr()[0] if scheduler else lr,
        )
    return model
