"""Implementation of a command to visualise UMAP offline."""

import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from torch import Tensor
from umap import UMAP

from cli.commands.base import BaseCommand
from src.data.image import ImageDataModule
from src.models.utils import make_model
from src.systems import ClassificationSystem
from src.systems.utils import collate_batches


class VisualiseUMAP(BaseCommand):
    """Visualise a UMAP of a neural network on a dataset."""

    def run(self) -> None:
        """Visualise a model embeddings on a dataset."""
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
            shuffle=False,
            batch_size=self.options.get("batch_size"),
            num_workers=self.options.get("num_workers"),
        )
        system = ClassificationSystem(model)

        # Predict the outputs on the specified dataloader.
        dm.setup()
        dataloader = getattr(dm, f"{self.options['split']}_dataloader")()
        batches = trainer.predict(system, dataloader)
        assert isinstance(batches, list), "output of predict cycle must be of type list"
        out = collate_batches(batches)

        _, predictions = torch.max(out["outputs"], 1)

        # Save the figure.
        fig = self.visualise_embeddings(
            out["embeddings"],
            predictions,
            out["labels"],
            dataloader.dataset.labels,
        )
        model_name = os.path.basename(self.options.get("network"))
        save_path = f'{model_name}_{self.options.get("dataset")}.jpg'
        os.makedirs(self.options.get("images_dir"), exist_ok=True)
        save_path = os.path.join(self.options.get("images_dir"), save_path)
        fig.savefig(save_path)
        self.console.print(f"image saved at {save_path}")

    @staticmethod
    def visualise_embeddings(
        embeddings: Tensor,
        predictions: Tensor,
        labels: Tensor,
        class_labels: List[str],
        random_state: int = 42,
    ) -> matplotlib.figure.Figure:
        """
        Apply UMAP to visualise embeddings. The method creates two figures, one
        containing the predictions and the other the ground-truth labels.

        :param embeddings: embeddings of the input data
        :param predictions: output predictions of the network
        :param labels: ground-truth labels
        :param class_labels: names of the classes
        """

        if isinstance(embeddings, Tensor):
            embeddings = embeddings.cpu().numpy()

        if isinstance(predictions, Tensor):
            predictions = predictions.cpu().numpy()

        if isinstance(labels, Tensor):
            labels = labels.cpu().numpy()

        trans = UMAP(n_neighbors=24, min_dist=1, random_state=random_state).fit(embeddings)
        fig = plt.figure(figsize=(20, 8), dpi=120)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(
            trans.embedding_[:, 0],
            trans.embedding_[:, 1],
            s=10,
            c=predictions,
            cmap="tab10",
        )
        ax1.set_title("Predictions")
        ax2 = fig.add_subplot(1, 2, 2)
        scatter = ax2.scatter(
            trans.embedding_[:, 0], trans.embedding_[:, 1], s=10, c=labels, cmap="tab10"
        )
        legend = ax2.legend(
            scatter.legend_elements()[0],
            class_labels,
            loc="lower left",
            title="Classes",
        )
        ax2.add_artist(legend)
        ax2.set_title("Labels")
        fig.tight_layout()
        return fig
