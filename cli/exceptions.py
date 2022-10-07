"""Module containing exceptions."""


class TerminationError(Exception):
    """Termination error exception class."""

    def __init__(self) -> None:
        super().__init__("External signal received: forcing termination")
