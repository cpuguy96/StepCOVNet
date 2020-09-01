import time


def timed_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        return_value = func(*args, **kwargs)
        end_time = time.time()
        print("Elapsed time was %g seconds for %s" % ((end_time - start_time), func.__name__))
        return return_value

    return wrapper


def timed(func, *args, **kwargs):
    start_time = time.time()
    return_value = func(*args, **kwargs)
    end_time = time.time()
    print("Elapsed time was %g seconds for %s" % ((end_time - start_time), func.__name__))
    return return_value
