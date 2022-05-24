import logging
import time

import math
from pytorch_lightning.callbacks import ProgressBarBase

TIME_INTERVALS = (
    ('w', 60 * 60 * 24 * 7),
    ('d', 60 * 60 * 24),
    ('h', 60 * 60),
    ('m', 60),
    ('s', 1),
)


def display_time(seconds, granularity=2):
    result = []

    seconds = int(round(seconds))

    for name, count in TIME_INTERVALS:
        value = seconds // count
        if value == 0 and len(result) == 0:
            continue
        seconds -= value * count
        if value == 1:
            name = name.rstrip('s')
        result.append(f"{value:02d}{name}")
    return ':'.join(result[:granularity])


class LoggingProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate):
        super().__init__()
        self._is_enable = True
        self._epoch_start_time = None
        self._validation_start_time = None
        self._train_start_time = None
        self._last_epoch_times = []
        self._validation_times = []
        self.refresh_rate = refresh_rate

    def disable(self):
        self._is_enable = False

    def enable(self):
        self._is_enable = True

    def _should_update(self, current: int, total: int) -> bool:
        return self._is_enable and (current % self.refresh_rate == 0 or current == total)

    def on_train_start(self, trainer, pl_module):
        logging.info("Training starting.")

    def on_train_end(self, trainer, pl_module):
        logging.info("Training ending.")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int, unused: int = 0) -> None:
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
    def train_batch(message):
        logging.info(message)

    @staticmethod
    def _replace_metric_key(metric_key):
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
    def _format_metric_string(metrics_dict):
        metric_list = []
        skip_keys = {"v_num"}

        for key, v in metrics_dict.items():
            key = LoggingProgressBar._replace_metric_key(key)
            v = float(v)
            if math.isnan(v) or key in skip_keys:
                continue
            if key:
                metric_list.append(f"{key}={float(v):2.2f}")

        return ", ".join(metric_list)

    @staticmethod
    def _average_time(time_list):
        return int(round(sum(time_list) / len(time_list)))

    def _average_epoch_time(self):
        if len(self._last_epoch_times) == 0:
            return 0
        return self._average_time(self._last_epoch_times)

    def _average_validation_time(self):
        if len(self._validation_times) == 0:
            return 0
        return self._average_time(self._validation_times)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start_time = time.time()
        if self._train_start_time is None:
            self._train_start_time = self._epoch_start_time

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._last_epoch_times.append(time.time() - self._epoch_start_time)
        self._last_epoch_times = self._last_epoch_times[-3:]

    def on_validation_start(self, trainer, pl_module) -> None:
        self._validation_start_time = time.time()
        logging.info("Validating model...")

    def on_validation_end(self, trainer, pl_module) -> None:
        self._validation_times.append(time.time() - self._validation_start_time)
        self._validation_times = self._validation_times[-3:]
        logging.info(f"Validation complete. ({self._format_metric_string(self.get_metrics(trainer, pl_module))})")
