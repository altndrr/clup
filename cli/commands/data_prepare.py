"""Implementation of a command to prepare datasets."""

from itertools import product

from cli.commands.base import BaseCommand
from src.datasets import BENCHMARK_DATASETS, DATASETS


class DataPrepare(BaseCommand):
    """Prepare a dataset."""

    def run(self) -> None:
        """Prepare a dataset."""
        dataset = DATASETS.get(self.options.get("dataset"))

        if dataset is None:
            raise ValueError(f"dataset name not in {list(DATASETS.keys())}")

        kwargs = {}
        kwargs["data_dir"] = self.options.get("data_dir")
        kwargs["raw_data_dir"] = self.options.get("raw_data_dir")

        domains = dataset.available_domains if len(dataset.available_domains) > 0 else [""]
        combinations = product(domains, dataset.available_splits)
        for i, (domain, split) in enumerate(combinations):
            download = self.options.get("dataset") in BENCHMARK_DATASETS and i == 0
            if domain == "":
                dataset(split=split, prepare=True, download=download, **kwargs)
            else:
                dataset(
                    domain=domain,
                    split=split,
                    prepare=True,
                    download=download,
                    **kwargs,
                )
