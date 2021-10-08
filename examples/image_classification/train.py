import torch
import logging
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from utils.checkpointmanager import CheckpointManager
from utils.etaestimator import ETAEstimator
from utils.resultlogger import ResultLogger
from utils.metricscalculator import MetricsCalculator
from binary_torchinfo.torchinfo import summary
from bitorch.quantizations.base import Quantization


def train_model(
        model: Module,
        train_data: DataLoader,
        test_data: DataLoader,
        result_logger: ResultLogger,
        checkpoint_manager: CheckpointManager,
        eta_estimator: ETAEstimator,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        start_epochs: int = 0,
        epochs: int = 10,
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

    # some code for model visualization / storing of initial state
    images, _ = iter(train_data).next()
    images = images.to(device)
    result_logger.log_model(model, images)
    checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, 0, f"{model.name}_untrained")
    summary_str = summary(model, verbose=0, input_data=images, depth=10,
                          quantization_base_class=Quantization, device=device)
    logging.info(f"Model summary:\n{summary_str}")

    # initialization of eta estimator
    total_number_of_batches = (epochs - start_epochs) * (len(train_data) + len(test_data))
    eta_estimator.set_iterations(total_number_of_batches)
    current_number_of_batches = 0
    best_accuracy = 0.0

    metrics = MetricsCalculator()

    for epoch in range(start_epochs, epochs):
        eta_estimator.epoch_start()
        logging.info(f"\n-------------------------- epoch {epoch + 1} --------------------------")

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
                    category="Batches",
                    step=current_number_of_batches * len(x_train),
                    loss=metrics.avg_loss(),
                )

            if idx % log_interval == 0 and idx > 0:
                result_logger.tensorboard_results(
                    category="Batches",
                    step=current_number_of_batches * len(x_train),
                    accuracy=metrics.accuracy(),
                    recall=metrics.recall(),
                    precision=metrics.precision(),
                    f1=metrics.f1(),
                    top_5_accuracy=metrics.top_5_accuracy(),
                )
                logging.info(
                    f"Loss in epoch {epoch + 1} for batch {idx}: {metrics.avg_loss()}, batch acc: {metrics.accuracy()},"
                    f" current lr: {scheduler.get_last_lr() if scheduler else lr}, eta: {eta_estimator.eta()}")

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
