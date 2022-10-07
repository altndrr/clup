"""Module containing a script to parse arguments."""

import re
import string
import warnings

from docopt import docopt

from cli import ARGUMENTS
from cli import __doc__ as doc
from cli import __version__, config


def parse_arguments():
    """
    Wrapper function around docopt. It adds the following functionalities:
    1) split arguments between commands and options
    2) remove symbols from the names of arguments
    3) load default values from configuration file
    4) remove None arguments
    5) cast arguments to type or use custom functions
    6) remove unknown arguments
    """
    options = docopt(doc, version=__version__)

    # Split commands and options.
    is_command = lambda x: x[0] in string.ascii_letters
    commands = {key: options[key] for key in options if is_command(key)}
    options = {key: options[key] for key in options if not is_command(key)}

    # Stringify options.
    format_key = lambda x: re.sub(r"^(<|\-)*|(\>)*$", "", x)
    options = {format_key(key): options[key] for key in options}

    # Parse options.
    arguments = set(ARGUMENTS.keys() & options.keys())
    for key in arguments:
        arg = ARGUMENTS.get(key)
        value = options.get(key)

        # Load the default value for the argument.
        # Prioritise the value in the config file over the value in ARGUMENTS.
        default = config.find(key, root="args") or arg.get("default")

        if value is None and default is not None:
            options[key] = default
            value = default

        is_unset = lambda x: x is None or ((isinstance(x, bool)) and x is False)
        if key in options and is_unset(value) and default is None:
            del options[key]
            continue

        if value is not None and arg is not None:
            parse_fn = arg.get("parse_fn", arg.get("type"))
            assert parse_fn is not None, "either `parse_fn` or `type` must be defined"
            options[key] = parse_fn(value)
        elif value is not None and arg is None:
            warnings.warn(f"argument {key} has no explicit type or parse_fn")

    # Remove unknown arguments.
    unknown_keys = set(options.keys()) - set(ARGUMENTS.keys())
    for key in unknown_keys:
        warnings.warn(f"removed unkown argument {key}")
        del options[key]

    return commands, options
