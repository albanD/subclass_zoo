import torch
from torch import Tensor
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import (
    TestCase, run_tests, disable_gc, parametrize,
    instantiate_parametrized_tests
)

import functools
import contextlib

from utils import no_dispatch
from base_tensor import BaseTensor

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
        # TODO: this is probably wrong for lifting batched
        # tensor, but for autograd it's OK cuz it detaches
        # TODO: no_dispatch here is wrong
        with no_dispatch():
            return cls._make_subclass(cls, elem)

    def __repr__(self):
        return f"WrapperTensor({super().__repr__()}, {repr(self.elem)})"

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

        def unwrap(t):
            if isinstance(t, cls) and t.level == max_level:
                return t.elem
            elif isinstance(t, torch.Tensor):
                # implicitly lift to the current level
                # TODO: hmm is this gonna copy everything?
                return WrapperTensor(t, max_level)
            else:
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t, max_level)
            else:
                return t

        return tree_map(
            wrap,
            func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        )

def grad_and_value(func):
    @functools.wraps(func)
    def wrapper(input):
        with new_level() as level:
            assert isinstance(input, torch.Tensor)
            input = WrapperTensor(input, level)
            input.requires_grad_()
            output = func(input)
            grad_input, = torch.autograd.grad(output, input, create_graph=True, allow_unused=True)
            return unwrap(grad_input, level), unwrap(output, level)
    return wrapper

def grad(func):
    @functools.wraps(func)
    def wrapper(input):
        grad_input, _  = grad_and_value(func)(input)
        return grad_input
    return wrapper

class FunctorchTest(TestCase):
    def test_basic(self):
        x = torch.randn([])
        result = grad(torch.sin)(x)
        self.assertEqual(result, torch.cos(x))

    def test_grad_of_grad(self):
        x = torch.randn([])
        result = grad(grad(lambda x: x ** 3))(x)
        self.assertEqual(result, 3 * x ** 2)

if __name__ == '__main__':
    run_tests()
