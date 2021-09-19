from typing import Any, List
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    top_k_accuracy_score,
    f1_score,
)


class MetricsCalculator():
    """class for dynamic metrics calculation"""
    _prediction: List[Any] = []
    _ground_truth: List[Any] = []
    _prediction_argmax = None
    _total_loss = 0.0
    _index = 0

    def clear(self) -> None:
        """resets the classes attributes"""
        self._prediction_argmax = None
        self._prediction = []
        self._ground_truth = []
        self._total_loss = 0.0
        self._index = 0

    def update(self, prediction: torch.Tensor, ground_truth: torch.Tensor, loss: torch.Tensor) -> None:
        """stores predicted and ground truth labels for metric calculation. accumulates the passed loss for
        avg loss calculation.

        Args:
            prediction (torch.Tensor): predicted labels (as batch)
            ground_truth (torch.Tensor): ground truth lables (as batch)
            loss (torch.Tensor): loss value
        """
        self._prediction += prediction.tolist()
        self._ground_truth += ground_truth.tolist()
        self._total_loss += loss.item()
        self._prediction_argmax = None

    def _calculate_argmax(self) -> None:
        """computes argmax of predicted labels. this function is called lazily when metrics are requested and only
        if it was not already called since the last updated

        Raises:
            ValueError: thrown if no prediction or ground truth lables have been stored yet.
        """
        if not self._ground_truth or not self._prediction:
            raise ValueError("cannot compute metrics with no entered data!")
        self._prediction_argmax = np.argmax(self._prediction, axis=1)

    def accuracy(self) -> float:
        if self._prediction_argmax is None:
            self._calculate_argmax()
        return accuracy_score(self._ground_truth, self._prediction_argmax)

    def top_5_accuracy(self) -> float:
        if self._prediction_argmax is None:
            self._calculate_argmax()
        labels = list(range(len(self._prediction[0])))
        return top_k_accuracy_score(self._ground_truth, self._prediction, k=5, labels=labels)

    def precision(self) -> float:
        if self._prediction_argmax is None:
            self._calculate_argmax()
        return precision_score(self._ground_truth, self._prediction_argmax, average="macro", zero_division=0)

    def recall(self) -> float:
        if self._prediction_argmax is None:
            self._calculate_argmax()
        return recall_score(self._ground_truth, self._prediction_argmax, average="macro", zero_division=0)

    def f1(self) -> float:
        if self._prediction_argmax is None:
            self._calculate_argmax()
        return f1_score(self._ground_truth, self._prediction_argmax, average="macro", zero_division=0)

    def avg_loss(self) -> float:
        return self._total_loss / len(self._prediction)
