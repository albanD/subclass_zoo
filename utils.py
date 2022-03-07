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


# IDK if this is actually useful or not
def unmake_subclass(tensor):
    with no_dispatch():
        return torch.Tensor._make_subclass(torch.Tensor, tensor)

def fill_defaults(args, n, defaults_tail):
    """
    __torch_dispatch__ doesn't guarantee the number of arguments you are
    passed (e.g., defaulted arguments are not passed); but usually it is
    convenient to pad out the arguments list with defaults.  This function
    helps you do that.

    Args:
        args: the list of positional arguments passed to __torch_dispatch__
        n: the number of arguments you are expecting to get
        defaults_tail: default values for the arguments, starting from the
            end of the list

    Example:

        >>> fill_defaults([1, 2, 3], 5, [3, 4, 5])
        [1, 2, 3, 4, 5]
        >>> fill_defaults([1, 2, 3], 5, [None, None, None])
        [1, 2, 3, None, None]]
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i-n+len(defaults_tail)])
    return r
