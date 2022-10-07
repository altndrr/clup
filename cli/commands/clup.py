"""Implementation of a command for CluP systems."""

from typing import List, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from cli import config
from cli.commands.base import BaseCommand
from src.data.image import ImageDataModule
from src.datasets import DATASETS
from src.models.utils import make_model
from src.systems.classification import ClassificationSystem
from src.systems.clup import CluPSystem
from src.systems.utils import collate_batches, predict_classes


class CluP(BaseCommand):
    """Apply CluP on a network."""

    def run(self) -> None:
        model = make_model(
            self.options.get("network"),
            num_classes=len(DATASETS[self.options.get("dataset")].labels),
            weight_norm=self.options.get("weight_norm"),
            mlp_head=self.options.get("mlp_head", False),
        )

        # Set command-specific default values.
        self.options["labeled_samples_size"] = self.options.get("labeled_samples_size", 0.1)

        # Define the logger.
        logger: Union[bool, WandbLogger] = False
        if self.options.get("wandb"):
            logger = WandbLogger(
                name=self.options.get("name"),
                project=self.options.get("project"),
                save_dir="wandb",
            )

            # Watch the model and log the options.
            logger.watch(model, log="gradients", log_freq=100)
            logger.log_hyperparams(self.options)

        # Create the datamodule.
        dm = ImageDataModule(
            self.options.get("dataset"),
            self.options.get("data_dir"),
            batch_size=self.options.get("batch_size"),
            num_workers=self.options.get("num_workers"),
            return_index=True,
        )

        # Get the sampling trainer.
        sampling_trainer = Trainer(
            accelerator=self.options.get("accelerator"),
            callbacks=[RichModelSummary(), RichProgressBar()],
            deterministic=config.get("deterministic"),
            enable_checkpointing=False,
            gpus=self.options.get("gpus"),
            max_epochs=1,
            logger=False,
            precision=self.options.get("precision"),
            strategy=self.options.get("strategy"),
        )

        # Define the sampling system.
        teacher_model = make_model(self.options.get("teacher"))
        teacher_system = ClassificationSystem(teacher_model)

        # Predict the ouputs for the train set.
        dm.setup()
        batches = sampling_trainer.predict(teacher_system, dm.predict_dataloader())
        assert batches is not None, "output of predict cycle must be non-empty"
        out = collate_batches(batches)

        # Cluster multiple times the non-selected samples.
        subset_idxs: List[int] = []
        labeled_samples_mode_kw = self.options.get("labeled_samples_mode_kw", {})
        iter_clusters = labeled_samples_mode_kw.get("iter_clusters", 1)
        for i in range(iter_clusters):
            # Keep only samples not in subset_idxs.
            not_selected = list(set(out["indices"].to(torch.int).tolist()) - set(subset_idxs))
            self.console.print(f"Clustering {len(not_selected)} samples")
            out_copy = {key: value[not_selected] for key, value in out.items()}
            out_copy = predict_classes(
                out_copy,
                mode=self.options.get("labeled_samples_mode"),
                **labeled_samples_mode_kw,
            )

            # Select a subset of pseudo labeled samples by the teacher (class-wise).
            num_samples = self.options.get("labeled_samples_size")
            confidences = out_copy["pseudo_labels_confidences"]
            assert hasattr(dm.predict_set, "labels"), "predict set must have `labels` property"
            for class_idx, _ in enumerate(getattr(dm.predict_set, "labels")):
                mask = out_copy["pseudo_labels"] == class_idx
                class_confidences = confidences[mask]
                class_idxs = out_copy["indices"][mask]
                num_samples_class = int(num_samples * len(class_idxs))
                self.console.print(f"{num_samples_class=}")
                subset_idxs.extend(
                    class_idxs[torch.topk(class_confidences, num_samples_class)[1]]
                    .to(torch.int)
                    .tolist()
                )

            if "pseudo_labels" not in out:
                out["pseudo_labels"] = out_copy["pseudo_labels"]
                if "soft_pseudo_labels" in out_copy:
                    out["soft_pseudo_labels"] = out_copy["soft_pseudo_labels"]
            else:
                out["pseudo_labels"][not_selected] = out_copy["pseudo_labels"]
                if "soft_pseudo_labels" in out_copy:
                    out["soft_pseudo_labels"][not_selected] = out_copy["soft_pseudo_labels"]
        self.console.print(f"Selected a total of {len(subset_idxs)} samples")

        # Get the trainer, the datamodule and the system.
        trainer = Trainer(
            accelerator=self.options.get("accelerator"),
            callbacks=[RichModelSummary(), RichProgressBar()],
            deterministic=config.get("deterministic"),
            enable_checkpointing=self.options.get("checkpoints"),
            gpus=self.options.get("gpus"),
            max_epochs=self.options.get("epochs"),
            logger=logger,
            precision=self.options.get("precision"),
            strategy=self.options.get("strategy"),
            reload_dataloaders_every_n_epochs=1,
        )

        # Use hard or soft targets.
        criterion_kw = self.options.get("criterion_kw", {})
        criterion: torch.nn.Module = torch.nn.CrossEntropyLoss(**criterion_kw)
        labels = out["pseudo_labels"]
        if self.options.get("labeled_samples_mode") == "soft-purity":
            labels = out["soft_pseudo_labels"]
            criterion_kw["reduction"] = criterion_kw.get("reduction", "batchmean")
            criterion = torch.nn.KLDivLoss(**criterion_kw)

        system = CluPSystem(
            model,
            labels,
            labeled_samples=subset_idxs,
            criterion=criterion,
            pseudo_labeling_mode=self.options.get("labeled_samples_mode"),
            pseudo_labeling_mode_kw=self.options.get("labeled_samples_mode_kw", {}),
            **self.options,
        )

        # If required, freeze part of the system.
        if self.options.get("freeze"):
            system.set_requires_grad(self.options.get("freeze"), False)

        # Fit the system to the data.
        trainer.fit(system, dm)
