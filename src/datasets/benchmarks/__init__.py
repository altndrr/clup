"""Init of the module containing benchmark datasets."""

from typing import Dict, List, Type

from src.datasets.base import BaseDataset
from src.datasets.benchmarks.domain_net import DomainNet
from src.datasets.benchmarks.mnist import MNIST
from src.datasets.benchmarks.office_31 import Office31
from src.datasets.benchmarks.office_home import OfficeHome
from src.datasets.benchmarks.svhn import SVHN
from src.datasets.benchmarks.usps import USPS
from src.datasets.benchmarks.visda_2017 import VisDA2017

__all__: List[str] = ["MNIST", "SVHN", "USPS"]

BENCHMARK_DATASETS: Dict[str, Type[BaseDataset]] = {
    "mnist": MNIST,
    "svhn": SVHN,
    "usps": USPS,
    "domainnet": DomainNet,
    "office-31": Office31,
    "office-home": OfficeHome,
    "visda2017": VisDA2017,
}
