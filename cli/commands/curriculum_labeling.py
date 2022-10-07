"""Implementation of a command for curriculum labeling systems."""

from copy import deepcopy
from typing import Any, Dict, List, Union

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from cli import config
from cli.commands.base import BaseCommand
from src.data.image import ImageDataModule
from src.datasets import DATASETS
from src.models.utils import make_model, subset_state_dict
from src.systems.classification import ClassificationSystem
from src.systems.curriculum_labeling import CurriculumLabelingSystem
from src.systems.utils import collate_batches, predict_classes


class CurriculumLabeling(BaseCommand):
    """Perform curriculum labeling on a network."""

    def run(self) -> None:
        model = make_model(
            self.options.get("network"),
            num_classes=len(DATASETS[self.options.get("dataset")].labels),
            weight_norm=self.options.get("weight_norm"),
        )

        # Set command-specific default values.
        self.options["iter_labeled"] = self.options.get("iter_labeled", 1)
        self.options["iter_unlabeled"] = self.options.get("iter_unlabeled", 5)
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

        # Sample supervised subset.
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
        sampling_system = ClassificationSystem(teacher_model)

        # Predict the outputs for the train set.
        dm.setup()
        batches = sampling_trainer.predict(sampling_system, dm.predict_dataloader())
        assert batches is not None, "output of predict cycle must be non-empty"
        out = collate_batches(batches)

        # Cluster multiple times the non-selected samples.
        subset_idxs: List[int] = []
        labeled_samples_mode_kw = self.options.get("labeled_samples_mode_kw", {})
        # Keep only samples not in subset_idxs.
        out = predict_classes(
            out,
            mode=self.options.get("labeled_samples_mode"),
            **labeled_samples_mode_kw,
        )

        # Select a subset of pseudo labeled samples by the teacher (class-wise).
        num_samples = self.options.get("labeled_samples_size")
        confidences = out["pseudo_labels_confidences"]
        subset_idxs = []
        assert hasattr(dm.predict_set, "labels"), "predict set must have `labels` property"
        for class_idx, _ in enumerate(getattr(dm.predict_set, "labels")):
            mask = out["pseudo_labels"] == class_idx
            class_confidences = confidences[mask]
            class_idxs = out["indices"][mask]
            num_samples_class = int(num_samples * len(class_idxs))
            self.console.print(f"{num_samples_class=}")
            subset_idxs.extend(
                class_idxs[torch.topk(class_confidences, num_samples_class)[1]]
                .to(torch.int)
                .tolist()
            )
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
        criterion = torch.nn.CrossEntropyLoss(**self.options.get("criterion_kw", {}))

        system = CurriculumLabelingSystem(
            model,
            labeled_samples=subset_idxs,
            labels=out["pseudo_labels"],
            criterion=criterion,
            pseudo_labeling_mode=self.options.get("labeled_samples_mode"),
            pseudo_labeling_mode_kw=self.options.get("labeled_samples_mode_kw", {}),
            **self.options,
        )

        # If required, freeze part of the system.
        if self.options.get("freeze"):
            system.set_requires_grad(self.options.get("freeze"), False)

        # Fit the system to the data.
        total_iter = self.options.get("iter_labeled") + self.options.get("iter_unlabeled")
        initial_weights = deepcopy(system.state_dict())
        for i in range(total_iter):
            if i > 0:
                # Reset weights.
                system.encoder.load_state_dict(subset_state_dict(initial_weights, "encoder"))
                system.classifier.load_state_dict(subset_state_dict(initial_weights, "classifier"))

                trainer.fit_loop.max_epochs += self.options.get("epochs")
                trainer.callbacks[1] = RichProgressBar()  # ! fix for multiple fits when using rich

            # Train for an iteration.
            trainer.fit(system, dm)
