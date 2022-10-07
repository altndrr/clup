"""Implementation of a command to test classification systems."""

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar

from cli.commands.base import BaseCommand
from src.data.image import ImageDataModule
from src.models.utils import make_model
from src.systems import ClassificationSystem


class ClassificationTest(BaseCommand):
    """Test a neural network on a classification task."""

    def run(self) -> None:
        model = make_model(self.options.get("network"))

        # Get the trainer, the datamodule and the system.
        trainer = Trainer(
            accelerator=self.options.get("accelerator"),
            callbacks=[RichModelSummary(), RichProgressBar()],
            deterministic=True,
            gpus=self.options.get("gpus"),
            logger=False,
            precision=self.options.get("precision"),
            strategy=self.options.get("strategy"),
        )
        dm = ImageDataModule(
            self.options.get("dataset"),
            self.options.get("data_dir"),
            augment=False,
            batch_size=self.options.get("batch_size"),
            num_workers=self.options.get("num_workers"),
        )
        system = ClassificationSystem(model)

        out = trainer.test(system, dm, verbose=False)
        self.console.print(out[0])
