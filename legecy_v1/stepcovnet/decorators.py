import time
from collections.abc import Callable


def timed_function(func: Callable):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        return_value = func(*args, **kwargs)
        end_time = time.time()
        print(
            "Elapsed time was %g seconds for %s"
            % ((end_time - start_time), func.__name__)
        )
        return return_value

    return wrapper


def timed(func: Callable, *args, **kwargs):
    start_time = time.time()
    return_value = func(*args, **kwargs)
    end_time = time.time()
    print(
        "Elapsed time was %g seconds for %s" % ((end_time - start_time), func.__name__)
    )
    return return_value
