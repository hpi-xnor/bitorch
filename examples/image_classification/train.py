import logging
import time
from math import floor

import torch
from torch.nn import Module
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from utils.checkpointmanager import CheckpointManager
from utils.etaestimator import ETAEstimator
from utils.resultlogger import ResultLogger


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

    total_number_of_batches = (epochs - start_epoch) * (len(train_data) + len(test_data))
    eta_estimator.set_iterations(total_number_of_batches)
    current_number_of_batches = 0
    best_accuracy = 0.0

    btic = time.time()

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        logging.info(f"\n-------------------------- epoch {epoch + 1} --------------------------")

        model.train()
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

                epoch_loss += loss.item()
                result_logger.tensorboard_results(
                    category="Train/Batches",
                    batch_num=current_number_of_batches,
                    loss=epoch_loss / max(idx, 1)
                )

            if idx % log_interval == 0 and idx > 0:
                progress = floor((current_number_of_batches / total_number_of_batches) * 100000.0) / 1000.0
                speed_in_sample_per_s = train_data.batch_size / (time.time() - btic)
                lr = scheduler.get_last_lr() if scheduler else lr
                if isinstance(lr, list):
                    lr = lr[0]
                logging.info(
                    f"train epoch {epoch + 1:3d} batch {idx:5d} ({progress:03.3f}%) - loss: {epoch_loss / idx:2.2f}, "
                    f"lr: {lr:.7f} (speed: {speed_in_sample_per_s:5.1f} samples/s)")

            btic = time.time()

        epoch_loss /= len(train_data)

        if scheduler:
            scheduler.step()

        model.eval()
        test_loss = 0.0
        correct = 0.0
        correct_top5 = 0.0
        # now validate model with test dataset
        with torch.no_grad():
            for idx, (x_test, y_test) in enumerate(test_data):
                with eta_estimator:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)

                    y_hat = model(x_test)
                    test_loss += criterion(y_hat, y_test).item()

                    # determine count of correctly predicted labels
                    predictions = torch.argmax(y_hat, dim=1)
                    _, predictions_top5 = torch.topk(y_hat, 5, dim=1)
                    correct += torch.sum(y_test == predictions).item()
                    for idx, top5 in enumerate(predictions_top5):
                        correct_top5 += int(y_test[idx] in top5)
        test_loss /= len(test_data)
        batch_size = test_data.batch_size
        if batch_size is None:
            batch_size = 1
        accuracy = correct / (len(test_data) * batch_size)
        accuracy_top5 = correct_top5 / (len(test_data) * batch_size)

        result_logger.log_result(
            epoch=epoch + 1,
            lr=scheduler.get_last_lr() if scheduler else lr,
            train_epoch_loss=epoch_loss,
            test_epoch_loss=test_loss,
            test_top1_acc=accuracy,
            test_top5_acc=accuracy_top5)
        result_logger.tensorboard_results(
            category="Train",
            epoch=epoch + 1,
            loss=epoch_loss
        )
        result_logger.tensorboard_results(
            category="Test",
            epoch=epoch + 1,
            loss=test_loss,
            top1_acc=accuracy,
            top5_acc=accuracy_top5
        )
        checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, epoch)
        if accuracy > best_accuracy:
            logging.info("updating best model....")
            checkpoint_manager.store_model_checkpoint(model, optimizer, scheduler, epoch, f"{model.name}_best")
            best_accuracy = accuracy
    return model
