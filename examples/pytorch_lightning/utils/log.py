import logging
import time
from typing import Optional, Any, Dict, List, Union

import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ProgressBarBase
from pytorch_lightning.utilities.types import STEP_OUTPUT

logger = logging.getLogger(__name__)

TIME_INTERVALS = (
    ('w', 60 * 60 * 24 * 7),
    ('d', 60 * 60 * 24),
    ('h', 60 * 60),
    ('m', 60),
    ('s', 1),
)


def display_time(seconds: float, granularity: int = 2) -> str:
    result: List[str] = []

    seconds = int(round(seconds))

    for name, count in TIME_INTERVALS:
        value = seconds // count
        if value == 0 and len(result) == 0:
            continue
        seconds -= value * count
        result.append(f"{value:02d}{name}")
    return ':'.join(result[:granularity])


class LoggingProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int) -> None:
        super().__init__()
        self._is_enabled = True
        self._epoch_start_time: float = 0.0
        self._validation_start_time: float = 0.0
        self._train_start_time: float = 0.0
        self._last_epoch_times: List[float] = []
        self._validation_times: List[float] = []
        self.refresh_rate = refresh_rate
        self.log_debug("Logging training progress...")

    @staticmethod
    def log_debug(message: str) -> None:
        # logger.debug(message)
        print(message)

    @staticmethod
    def log_info(message: str) -> None:
        # logger.info(message)
        print(message)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self.log_debug(f"Logging setup... ( is root trainer: {trainer.is_global_zero} )")
        super().setup(trainer, pl_module, stage)

    def disable(self) -> None:
        self.log_debug("Logging disabled...")
        self._is_enabled = False

    def enable(self) -> None:
        self.log_debug("Logging enabled...")
        self._is_enabled = True

    def _should_update(self, current: int, total: Union[int, float]) -> bool:
        return self._is_enabled and (current % self.refresh_rate == 0 or current == int(total))

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_info("Starting training...")

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.log_info("Ending training.")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        if not self._should_update(self.train_batch_idx, self.total_train_batches):
            return

        percent = (self.train_batch_idx / self.total_train_batches) * 100

        time_in_this_epoch = time.time() - self._epoch_start_time
        epoch_total_est = int(round((time_in_this_epoch * self.total_train_batches) / self.train_batch_idx))
        eta_epoch = display_time(epoch_total_est - time_in_this_epoch)
        full_epochs_left = trainer.max_epochs - trainer.current_epoch
        if full_epochs_left < 0:
            full_epochs_left = 0
        if self._average_epoch_time() > 0:
            epoch_total_est = self._average_epoch_time() + self._average_validation_time()
        eta_train = display_time(epoch_total_est - time_in_this_epoch + full_epochs_left * epoch_total_est)

        epoch_info = f"Epoch {trainer.current_epoch:3d}"
        batch_info = f"{self.train_batch_idx:4d}/{self.total_train_batches:4d} ({percent:5.1f}%)"
        metrics = self._format_metric_string(self.get_metrics(trainer, pl_module))
        eta_info = f"ETA: {eta_epoch} & {eta_train}"
        self.train_batch(f"{epoch_info} - {batch_info} - {metrics} - {eta_info}")

    @staticmethod
    def train_batch(message: str) -> None:
        LoggingProgressBar.log_info(message)

    @staticmethod
    def _replace_metric_key(metric_key: str) -> str:
        remove_strings = [
            "metrics/",
            "/train",
            "train-",
            "/test",
            "test-",
        ]
        for s in remove_strings:
            metric_key = metric_key.replace(s, "")
        return metric_key.replace("accuracy", "acc")

    @staticmethod
    def _format_metric_string(metrics_dict: Dict[str, Union[int, str]]) -> str:
        metric_list = []
        skip_keys = {"v_num"}

        for key, value in metrics_dict.items():
            key = LoggingProgressBar._replace_metric_key(key)
            if key in skip_keys:
                continue
            try:
                f_value = float(value)
                if math.isnan(f_value):
                    continue
                if key:
                    metric_list.append(f"{key}={f_value:2.2f}")
            except ValueError:
                if key:
                    metric_list.append(f"{key}={value}")

        return ", ".join(metric_list)

    @staticmethod
    def _average_time(time_list: List[float]) -> int:
        return int(round(sum(time_list) / len(time_list)))

    def _average_epoch_time(self) -> int:
        if len(self._last_epoch_times) == 0:
            return 0
        return self._average_time(self._last_epoch_times)

    def _average_validation_time(self) -> int:
        if len(self._validation_times) == 0:
            return 0
        return self._average_time(self._validation_times)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._epoch_start_time = time.time()
        if self._train_start_time is None:
            self._train_start_time = self._epoch_start_time

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._last_epoch_times.append(time.time() - self._epoch_start_time)
        self._last_epoch_times = self._last_epoch_times[-3:]

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._validation_start_time = time.time()
        self.log_info("Validating model...")

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._validation_times.append(time.time() - self._validation_start_time)
        self._validation_times = self._validation_times[-3:]
        self.log_info(f"Validation complete. ({self._format_metric_string(self.get_metrics(trainer, pl_module))})")
