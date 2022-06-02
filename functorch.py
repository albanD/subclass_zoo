import contextlib

import functools

import torch
from base_tensor import BaseTensor
from torch import Tensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map
from torch.overrides import enable_reentrant_dispatch

from utils import no_dispatch

# TODO: batched tensor (metadata doesn't match, so this needs more APIs)

LEVEL = 0


@contextlib.contextmanager
def new_level():
    global LEVEL
    LEVEL += 1
    try:
        yield LEVEL
    finally:
        LEVEL -= 1


def unwrap(t, level):
    if isinstance(t, WrapperTensor) and t.level == level:
        return t.elem
    else:
        return t


class WrapperTensor(BaseTensor):
    @staticmethod
    def __new__(cls, elem, level):
        # This is probably wrong for batched tensor, for autograd
        # it's good because make_subclass internally detaches.
        # no_dispatch is required to prevent detach form going to subclass.
        with no_dispatch():
            return cls._make_subclass(cls, elem)

    def __repr__(self):
        return f"WrapperTensor{self.level}({super().__repr__()}, {repr(self.elem)})"

    def __init__(self, elem, level):
        self.elem = elem
        self.level = level

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        max_level = -1

        def find_level(t):
            nonlocal max_level
            if isinstance(t, cls):
                max_level = max(max_level, t.level)

        # TODO: don't use tree_map
        tree_map(find_level, args)
        tree_map(find_level, kwargs)

        def matches_level(t):
            return isinstance(t, cls) and t.level == max_level

        def unwrap(t):
            if matches_level(t):
                return t.elem
            else:
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not matches_level(t):
                return cls(t, max_level)
            else:
                return t

        with enable_reentrant_dispatch():
            return tree_map(
                wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            )


def grad_and_value(func):
    @functools.wraps(func)
    def wrapper(input):
        with new_level() as level:
            assert isinstance(input, torch.Tensor)
            input = WrapperTensor(input, level)
            input.requires_grad_()
            output = func(input)
            (grad_input,) = torch.autograd.grad(
                output, input, create_graph=True, allow_unused=True
            )
            return unwrap(grad_input, level), unwrap(output, level)

    return wrapper


def grad(func):
    @functools.wraps(func)
    def wrapper(input):
        grad_input, _ = grad_and_value(func)(input)
        return grad_input

    return wrapper


class FunctorchTest(TestCase):
    def test_basic(self):
        x = torch.randn([])
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_grad_of_grad(self):
        x = torch.randn([])
        result = grad(grad(lambda x: x**3))(x)
        self.assertEqual(result, 6 * x)


if __name__ == "__main__":
    run_tests()
