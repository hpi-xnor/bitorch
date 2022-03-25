from pytorch_lightning import LightningModule


class ModelWrapper(LightningModule):
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def training_step(self, batch, batch_idx):
        ...

    def validation_step(self, batch):
        ...

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.scheduler,
        }
