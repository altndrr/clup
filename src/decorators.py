"""Module containing decorators."""

import multiprocessing
from functools import partial, wraps
from typing import Any, Callable, Optional

from pandas import DataFrame


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


def multiprocess(
    func: Optional[Callable[..., Any]] = None,
    data_var: str = "data",
    workers: int = 4,
    **kwargs,
) -> Any:
    """
    Split the work on a list or pandas.DataFrame between multiple processes.

    :param func: function to wrap
    :param data_var: name of the variable containing the data to split
    :param workers: number of workers to use to process the data
    """
    if func is None:
        return partial(multiprocess, data_var=data_var, workers=workers, **kwargs)

    @wraps(func)
    def inner(*args, **inner_kwargs):
        if data_var not in inner_kwargs:
            raise ValueError(f"{data_var} is not in **kwargs")

        data_items = inner_kwargs[data_var]
        assert isinstance(
            data_items, (list, DataFrame)
        ), "`data_items` must be of type list or DataFrame"

        # Evaluate the number of samples per process.
        size = len(data_items)
        item_residual = size % workers
        items_per_process = (size - item_residual) / workers
        items_per_process = int(items_per_process)

        # Instantiate the processes with their custom arguments.
        if len(data_items) > 1:
            processes = []
            for i in range(workers):
                # Get the partition of data for the current process.
                start_index = items_per_process * i
                end_index = start_index + items_per_process
                sub = data_items[start_index:end_index]

                # Copy the kwargs and overwrite the data variable.
                process_kwargs = inner_kwargs.copy()
                process_kwargs[data_var] = sub

                # Store the new process.
                processes.append(
                    multiprocessing.Process(target=func, args=args, kwargs=process_kwargs)
                )

            # Start each process.
            for process in processes:
                process.start()

            # Wait for each process to complete.
            for process in processes:
                process.join()

        # If necessary, create another process for the residual data.
        if item_residual != 0:
            start_index = items_per_process * workers
            sub = data_items[start_index:]
            process_kwargs = inner_kwargs.copy()
            process_kwargs[data_var] = sub

            assert func is not None, "`func` must be not None"
            func(*args, **process_kwargs)

    return inner
