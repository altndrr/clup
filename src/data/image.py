"""Module containing image-based datamodule."""

from typing import Dict, List, Optional

from torch.utils.data.dataloader import default_collate
from torchvision import transforms as T

from src.data.base import BaseDataModule
from src.data.utils import id_collate
from src.datasets import DATASETS


class ImageDataModule(BaseDataModule):
    """Implementation of a ImageFolder-based datamodule."""

    def __init__(
        self,
        name: str,
        data_dir: str,
        *args,
        prepare: bool = False,
        augment: bool = True,
        batch_size: int = 64,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, **kwargs
        )

        dataset = DATASETS.get(name)
        if dataset is None:
            raise ValueError(f"dataset not in {list(DATASETS.keys())}")

        self.dataset = dataset
        self.prepare = prepare
        self.augment = augment

        self.dataset_kwargs: Dict = {}
        return_index = self.kwargs.get("return_index", False)
        self.dataset_kwargs["data_dir"] = data_dir
        self.dataset_kwargs["raw_data_dir"] = self.kwargs.get("raw_data_dir")
        self.dataset_kwargs["return_index"] = return_index
        self.dataset_kwargs["collate_fn"] = id_collate if return_index else default_collate

    def prepare_data(self) -> None:
        if not self.prepare:
            return

        if not hasattr(self.dataset, "prepare_data"):
            return

        for split in self.dataset.available_splits:
            self.dataset(split=split, prepare=True, **self.dataset_kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        train_set = self.dataset(
            split="train",
            transform=T.Compose([*self.augmentations, *self.transforms]),
            **self.dataset_kwargs,
        )
        test_set = self.dataset(
            split="test", transform=T.Compose([*self.transforms]), **self.dataset_kwargs
        )

        self.train_set = train_set
        self.val_set = test_set
        self.test_set = test_set
        self.predict_set = train_set

        if "val" in self.dataset.available_splits:
            valid_set = self.dataset(
                split="val", transform=T.Compose([*self.transforms]), **self.dataset_kwargs
            )
            self.val_set = valid_set

    @property
    def augmentations(self) -> List:
        """Returns the extra augmentations on the inputs."""
        if not self.augment:
            return []

        return [
            T.RandomHorizontalFlip(),
            T.RandomAffine(35, translate=(0.15, 0.15)),
            T.ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.5, 1.5),
                hue=(-0.5, 0.5),
            ),
        ]

    @property
    def transforms(self) -> List:
        """Returns the default transformations of the inputs."""
        return [
            T.Resize(160),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
