from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class CustomWandbLogger(WandbLogger):
    """
    Customized Wandb Logger with the following changes:

        - the last model is not uploaded to wandb at the end of the training
    """

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._checkpoint_callback:
            # disable saving the last model to wandb
            self._checkpoint_callback.last_model_path = ""
        super().finalize(status)
