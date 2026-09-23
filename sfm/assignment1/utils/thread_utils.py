import multiprocessing
import threading


def run_on_thread(function):
    def wrap(*args, **kwargs):
        def target():
            try:
                function(*args, **kwargs)
            except Exception as error:
                # Keep background failures available to the GUI instead of
                # treating a stopped worker as a successful reconstruction.
                t.exception = error

        t = threading.Thread(target=target, daemon=True)
        t.exception = None
        t.start()

        return t
    return wrap


def run_on_process(function):
    def wrap(*args, **kwargs):
        process = multiprocessing.Process(target=function, args=args, kwargs=kwargs)
        process.start()
        return process 

    return wrap
