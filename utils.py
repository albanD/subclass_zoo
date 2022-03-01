import contextlib
import torch

# Dumping ground for utilities that should eventual make their way into
# PyTorch proper


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard
