"""Module containing decorators."""

from functools import partial, wraps
from typing import Any, Callable, Optional


def catch(func: Optional[Callable[..., Any]] = None, raise_unhandled: bool = True, **kwargs) -> Any:
    """
    Catch exceptions (ending with 'Error') and react to them.

    :param func: function to wrap
    :param raise_unhandled: if exception not handled, re-raise.
    """
    if func is None:
        return partial(catch, raise_unhandled=raise_unhandled, **kwargs)

    @wraps(func)
    def inner(*args, **inner_kwargs) -> None:
        try:
            assert func is not None, "`func` must be not None"
            func(*args, **inner_kwargs)
        except Exception as error:  # pylint: disable=broad-except
            error_name = str(type(error).__name__).lower()
            handled = False

            # Forward the exception if it is not a "*Error".
            if not error_name.endswith("error"):
                raise error

            # Dynamically handle the exception with the corresponding function.
            error_name = error_name[:-5]
            for key, catch_func in kwargs.items():
                if error_name == key:
                    handled = True

                    if callable(catch_func):
                        catch_func()

            # Forward the exception if no match was found and handling was required.
            if not handled and raise_unhandled:
                raise error

    return inner
