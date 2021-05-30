import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Adam
import logging


def train_model(
        model: Module,
        train_data: DataLoader,
        test_data: DataLoader,
        criterion: _Loss,
        epochs: int = 10,
        lr: float = 0.001,
        log_interval: int = 100,
        gpu: bool = False) -> Module:
    model.to('cuda' if gpu else 'cpu')

    optimizer = Adam(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0

        model.train()
        for idx, (x_train, y_train) in enumerate(train_data):
            optimizer.zero_grad()
            x_train.to('cuda' if gpu else 'cpu')
            y_train.to('cuda' if gpu else 'cpu')

            y_hat = model(x_train)
            loss = criterion(y_hat, y_train)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if idx % log_interval == 0 and idx > 0:
                logging.info(f"    Loss in epoch {epoch + 1} for batch {idx}: {epoch_loss / idx}")
        epoch_loss /= len(train_data)

        model.eval()
        test_loss = 0.0
        correct = 0.0
        # now validate model with test dataset
        with torch.no_grad():
            for idx, (x_test, y_test) in enumerate(test_data):
                x_test.to('cuda' if gpu else 'cpu')
                y_test.to('cuda' if gpu else 'cpu')

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
            f"Epoch {epoch + 1} train loss: {epoch_loss}, test lo# type: ignoress: {test_loss}, test accuracy: {accuracy}")

    return model
