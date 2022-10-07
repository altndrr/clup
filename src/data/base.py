"""Module containing an abstract base datamodule."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class BaseDataModule(ABC, pl.LightningDataModule):
    """Implementation of an abstract class for datamodules."""

    def __init__(
        self, *args, batch_size: int = 64, num_workers: int = 0, shuffle: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set: Dataset
        self.val_set: Dataset
        self.test_set: Dataset
        self.predict_set: Dataset

        self.shuffle = shuffle

        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError("method is not implemented")

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError("method is not implemented")

    def train_dataloader(self) -> Union[List[DataLoader], DataLoader]:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> Union[List[DataLoader], DataLoader]:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> Union[List[DataLoader], DataLoader]:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Union[List[DataLoader], DataLoader]:
        return DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
