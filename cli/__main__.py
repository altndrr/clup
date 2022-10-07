"""Main script of the cli module."""

import logging
import os
from bdb import BdbQuit

from pytorch_lightning import seed_everything
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_traceback

import wandb
from cli import COMMANDS, config
from cli.exceptions import TerminationError
from cli.parser import parse_arguments
from cli.signals import handle_sigterm

if __name__ == "__main__":
    console = Console(record=True)

    handle_sigterm()
    install_traceback()
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logging.captureWarnings(True)

    try:
        commands, options = parse_arguments()
        console.log(f"[bold]Options[/bold]: {options}")

        # Register the environmental variables and remove them from the
        # configuration file.
        if config.get("environment"):
            for name, value in config.get("environment").items():
                os.environ[name] = str(value)
            config.remove("environment")

        # If needed, fix seeds and backends to guarantee reproducibility.
        if config.get("deterministic"):
            logging.getLogger("pytorch_lightning.utilities.seed").disabled = True
            console.log(f"Global seed set to {seed_everything(1234)}")

        # Dynamically match the command the user with a pre-defined command class.
        executed = False
        for key, value in commands.items():
            command = COMMANDS.get(key)
            if command and value:
                cmd = command(options, console)
                cmd.run()
                executed = True
                break

        if not executed:
            console.log("[red]No match found for command[/red]")
    except BdbQuit:
        console.log("Closing debugger")
    except (KeyboardInterrupt, TerminationError):
        console.log("Termination signal received")
    finally:
        wandb.finish()

    os.makedirs("./media/html/", exist_ok=True)
    console.save_html("./media/html/console.html")
