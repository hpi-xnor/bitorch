from argparse import Namespace
from typing import Any

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only


class CustomWandbLogger(WandbLogger):
    """
    Customized Wandb Logger with the following changes:

        - the last model is not uploaded to wandb at the end of the training
        - automatically adds some tags based on the command line arguments
    """

    def __init__(self, script_args: Namespace, *args: Any, **kwargs: Any) -> None:
        kv_tags = ["model", "dataset"]
        wandb_tags = [f"{k}:{getattr(script_args, k, 'unknown')}" for k in kv_tags]
        if script_args.dev_run:
            wandb_tags.append("dev-run")
        if script_args.teacher:
            wandb_tags.append("kd")
        if "tags" in kwargs:
            kwargs["tags"].extend(wandb_tags)
        else:
            kwargs["tags"] = wandb_tags
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._checkpoint_callback:
            # disable saving the last model to wandb
            self._checkpoint_callback.last_model_path = ""
        super().finalize(status)
