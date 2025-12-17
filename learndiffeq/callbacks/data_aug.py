# Data Augmentation training callback

# Libraries
from pytorch_lightning.callbacks import Callback

class AdaptiveDataset(Callback):
    def on_train_epoch_start(self, trainer, pl_module, *args, **kwargs):
        trainer.datamodule.refresh_dataset(pl_module)
