"""Implementation of an abstract class for commands."""

from abc import ABC, abstractmethod
from typing import Dict

from rich.console import Console


class BaseCommand(ABC):
    """A base command."""

    def __init__(self, options: Dict, console: Console, *args, **kwargs) -> None:
        self.options = options
        self.console = console
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def run(self) -> None:
        """Run the command"""
        raise NotImplementedError("method is not implemented")
