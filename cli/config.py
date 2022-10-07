"""Module containing configuration values."""

from typing import Any, Optional

import toml

_CONFIGS = toml.load("config.toml")


def keys():
    """Get the list of keys."""
    return _CONFIGS.keys()


def values():
    """Get the list of values."""
    return _CONFIGS.values()


def items():
    """Get the list of items."""
    return _CONFIGS.items()


def get(key: str):
    """
    Get value in dictionary given its glob key.

    :param key: key of the value
    """
    key_parts = key.split("/")

    value: Any = _CONFIGS
    for part in key_parts:
        value = value.get(part)

    return value


def find(key: str, root: Optional[str] = None):
    """
    Search recursively for a key in the config file.
    Return the value of the key or None if not found.

    :param key: key to search
    """
    collection: Any = _CONFIGS

    if root:
        collection = collection.get(root)

    for name, val in collection.items():
        if name == key:
            return val

        if isinstance(val, dict):
            found = _find(key, val)
            if found:
                return found

    return None


def _find(key: str, collection: dict):
    """
    Search recursively for a key in a collection.
    Return the value of the key or None if not found.

    :param key: key to search
    :param collection: dictionary to search into
    """
    for name, val in collection.items():
        if name == key:
            return val

        if isinstance(val, dict):
            found = _find(key, val)
            if found:
                return found

    return None


def remove(key: str):
    """
    Remove an item given its glob key.

    :param key: key of the value
    """
    key_parts = key.split("/")
    last_key = key_parts.pop(-1)

    value: Any = _CONFIGS
    for part in key_parts:
        value = value.get(part)

    del value[last_key]
