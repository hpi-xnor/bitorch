# type: ignore
import pytorch_lightning as pl

from bitorch.quantizations import ProgressiveSign
from bitorch.quantizations.progressive_sign import config as progressive_sign_config


class ProgressiveSignScalerCallback(pl.callbacks.Callback):
    """Callback that updates the scale of progressive sign functions based on current epoch."""

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        scale = trainer.current_epoch / trainer.max_epochs
        progressive_sign_config.progressive_sign_scale = scale
        for logger in trainer.loggers:
            logger.log_metrics(
                {
                    "_progressive_sign_scale": scale,
                    "_progressive_sign_temperature": ProgressiveSign.default_transform(scale),
                },
                step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )
