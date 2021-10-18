import logging
from typing import Tuple, Union
import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path


class CheckpointManager():
    """Stores and loades training checkpoints. Maintains a fixed number of checkpoints, deletes older ones"""

    def __init__(self, checkpoint_store_dir: str, keep_count: int) -> None:
        """creates storing directory

        Args:
            checkpoint_store_dir (str): path to storing directory
            keep_count (int): number of checkpoints to keep
        """
        if checkpoint_store_dir:
            self._store_dir = Path(checkpoint_store_dir)
            self._store_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._store_dir = None  # type: ignore
            logging.warning("No checkpoint store dir given, checkpoints will not be stored!")
        self._keep_count = keep_count

    def store_model_checkpoint(
            self,
            model: Module,
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
            epoch: int,
            checkpoint_name: Union[str, None] = None) -> None:
        """stores the model, optimizer and lr scheduler state dicts and the epoch count. Deletes oldest checkpoints

        Args:
            model (Module): Model to store state dict of
            optimizer (Optimizer): optimizer to store
            lr_scheduler (_LRScheduler): lr scheduler to store
            epoch (int): epoch count
            checkpoint_name (Union[str, None], optional): special name of checkpoint, e.g. for storing the best model.
                if omitted, a generic name will be used. Defaults to None.
        """

        if not self._store_dir:
            return

        if not checkpoint_name:
            checkpoint_path = self._store_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth"
        else:
            checkpoint_path = self._store_dir / f"{checkpoint_name}.pth"
        logging.debug(f"storing checkpoint {checkpoint_path}...")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None
        }, checkpoint_path)

        if self._keep_count > 0:
            existing_checkpoints = list(self._store_dir.iterdir())
            sorted_checkpoints = sorted(existing_checkpoints, key=lambda checkpoint: checkpoint.stat().st_ctime)
            checkpoints_to_delete = sorted_checkpoints[:-self._keep_count]

            logging.debug(f"deleting checkpoints: {checkpoints_to_delete}")
            for checkpoint in checkpoints_to_delete:
                checkpoint.unlink()

    def load_checkpoint(
            self,
            path: str,
            model: Module,
            optimizer: Optimizer,
            lr_scheduler: _LRScheduler,
            pretrained: bool = False) -> Tuple[Module, Optimizer, _LRScheduler, int]:
        """loads the checkpoint at the given path. restores model, optimizer and lr scheduler state dict. !Note!: the model,
        optimizer and lr scheduler have to be the same as the stored ones, e.g. a resnet cannot load the state dict of a
        lenet obviously. only loads model state dict if fresh start is activated.

        Args:
            path (str): path to checkpoint
            model (Module): model to load the state dict
            optimizer (Optimizer): optimizer to load the state dict
            lr_scheduler (_LRScheduler): lr scheduler to load the state dict
            pretrained (bool, optional): toggles use of pretrained model, i.e. if activated, the optimizer and lr
                scheduler are not modified and epoch will be set to 0. Defaults to False.

        Raises:
            ValueError: thrown if checkpoint at given path does not exist.

        Returns:
            Tuple[Module, Optimizer, _LRScheduler, int]: the model with the loaded state dict, the optimizer and lr
                scheduler (with loaded state dict or unmodified) and epoch (from checkpoint or 0s)
        """

        if not path or not Path(path).exists():
            raise ValueError("checkpoint loading path not given or not existing!")
        logging.debug(f"loading checkpoint {path}....")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        if pretrained:
            epoch = 0
            logging.info(f"using a pretrained model from checkpoint {path} ...")
        else:
            epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lr_scheduler and checkpoint["lr_scheduler"]:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            logging.info(f"resuming model training from epoch {epoch}, loaded from checkpoint {path}...")

        return model, optimizer, lr_scheduler, epoch
