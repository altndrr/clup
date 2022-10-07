"""Module to handle signals."""

import signal
from types import FrameType
from typing import Optional

from cli.exceptions import TerminationError


def handle_sigterm() -> None:
    """Register handlers to manage termination signals."""

    def __handle_sigterm(signum: int, frame: Optional[FrameType]):
        raise TerminationError()

    signal.signal(signal.SIGINT, __handle_sigterm)
    signal.signal(signal.SIGINT, __handle_sigterm)
