"""Module containing a datamodule to collate many datamodules together."""

from typing import Any, List, Optional, Union

from torch.utils.data import DataLoader

from src.data.base import BaseDataModule


class CollateDataModule(BaseDataModule):
    """Implementation of an abstract class for datamodules."""

    def __init__(self, *dms: List[BaseDataModule]) -> None:
        super().__init__()
        self.dms = dms

        if len(dms) < 2:
            raise ValueError("class must contain >= 2 datamodules")

    def prepare_data(self) -> None:
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dm.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dm.setup()

    def train_dataloader(self) -> Union[List[DataLoader[Any]], DataLoader[Any]]:
        dms: List[DataLoader] = []
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dms.extend(*dm.train_dataloader())

        return dms

    def val_dataloader(self) -> Union[List[DataLoader[Any]], DataLoader[Any]]:
        dms: List[DataLoader] = []
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dms.extend(*dm.val_dataloader())

        return dms

    def test_dataloader(self) -> Union[List[DataLoader[Any]], DataLoader[Any]]:
        dms: List[DataLoader] = []
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dms.extend(*dm.test_dataloader())

        return dms

    def predict_dataloader(self) -> Union[List[DataLoader[Any]], DataLoader[Any]]:
        dms: List[DataLoader] = []
        for dm in self.dms:
            assert isinstance(dm, BaseDataModule), "`dm` must be of class BaseDataModule"
            dms.extend(*dm.predict_dataloader())

        return dms
