import torch
from torch.utils._pytree import PyTree, tree_flatten, tree_unflatten

import contextlib
from typing import Any

# Dumping ground for utilities that should eventual make their way into
# PyTorch proper


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def tree_map2(fn: Any, pytree1: PyTree, pytree2: PyTree) -> PyTree:
    flat_args1, spec1 = tree_flatten(pytree1)
    flat_args2, spec2 = tree_flatten(pytree2)
    assert spec1 == spec2
    return tree_unflatten(
        [fn(i, j) for i, j in zip(flat_args1, flat_args2)], spec1)
