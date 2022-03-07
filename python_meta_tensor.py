import torch
from torch.utils._pytree import tree_map, tree_flatten
from torch.testing._internal.common_utils import (
    TestCase, run_tests
)
from torch.utils._python_dispatch import enable_python_mode

import torch.nn

import itertools

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
    """
    if n - len(defaults_tail) > len(args):
        raise RuntimeError("not enough defaults to fill arguments")
    r = list(args)
    for i in range(len(args), n):
        r.append(defaults_tail[i-n+len(defaults_tail)])
    return r

class PythonMetaTensorMode(torch.Tensor):
    # TODO: figure out a better idiom for this; "pure" modes shouldn't be
    # instantiated so arguably they shouldn't be torch.Tensor subclasses,
    # but then making sure someone doesn't actually try to instantiate this
    # causes mixins on the tensor itself to stop working
    @staticmethod
    def __new__(cls, elem):
        raise RuntimeError("this mode mixin cannot actually be instantiated")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Only interpose for meta invocations
        flat_args, _ = tree_flatten(args)
        flat_kwargs, _ = tree_flatten(kwargs)
        if not any(isinstance(t, torch.Tensor) and t.is_meta for t in itertools.chain(flat_args, flat_kwargs)):
            return super().__torch_dispatch__(func, types, args, kwargs)

        if func == torch.ops.aten._embedding_bag:
            # Defaults can be determined by reading native_functions.yaml
            # We will soon make these available directly from the torch.ops
            # API, waiting on https://github.com/pytorch/pytorch/pull/72673
            args = fill_defaults(args, 9, [False, 0, False, None, False, -1])
            weight, indices, offsets, scale_grad_by_freq, mode, sparse, \
                per_sample_weights, include_last_offset, padding_idx = args
            # I determined the meaning of the outputs and sizes by reading
            # over the kernel in aten/src/ATen/native/EmbeddingBag.cpp
            output = weight.new_empty(
                offset.size(0) - 1 if include_last_offset else offsets.size(0),
                weight.size(1)
            )
            MODE_SUM, MODE_MEAN, MODE_MAX = range(3)
            if mode == MODE_MEAN or mode == MODE_MAX:
                offset2bag = offsets.new_empty(indices.size(0))
            else:
                offset2bag = offsets.new_empty(0)
            bag_size = offsets.new_empty(offsets.size())
            max_indices = offsets.new_empty(bag_size.size())
            return output, offset2bag, bag_size, max_indices
        # add your other patches here

        return super().__torch_dispatch__(func, types, args, kwargs)

class PythonMetaTensor(PythonMetaTensorMode):
    @staticmethod
    def __new__(cls, elem):
        # TODO: this will not backprop correctly (as all requires grad inputs
        # will look like leaves) but it will "look" like it has correct
        # requires_grad.  Once https://github.com/pytorch/pytorch/pull/73850
        # lands you can delete this static method entirely
        return cls._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem):
        assert elem.is_meta

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Propagate the wrapper
        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, super().__torch_dispatch__(func, types, args, kwargs))


class TrivialTensorTest(TestCase):
    def test_basic(self):
        x = PythonMetaTensor(torch.empty(2, 2, device='meta'))
        y = x + x
        self.assertEqual(y.shape, (2, 2))

    def test_embedding_bag(self):
        embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum', device='meta')
        input = torch.empty(8, dtype=torch.long, device='meta')
        offsets = torch.empty(2, dtype=torch.long, device='meta')
        self.assertRaises(NotImplementedError, lambda: embedding_sum(input, offsets))
        r = embedding_sum(PythonMetaTensor(input), PythonMetaTensor(offsets))
        self.assertEqual(r, torch.empty((2, 3), dtype=torch.float, device='meta'))

    def test_embedding_bag_via_mode(self):
        with enable_python_mode(PythonMetaTensorMode):
            embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum', device='meta')
            input = torch.empty(8, dtype=torch.long, device='meta')
            offsets = torch.empty(2, dtype=torch.long, device='meta')
            r = embedding_sum(input, offsets)
            self.assertEqual(r, torch.empty((2, 3), dtype=torch.float, device='meta'))

            # Make sure we don't interpose on non-meta computation
            embedding_sum = torch.nn.EmbeddingBag(10, 3, mode='sum')
            input = torch.tensor([1,2,4,5,4,3,2,9], dtype=torch.long)
            offsets = torch.tensor([0,4], dtype=torch.long)
            r = embedding_sum(input, offsets)
            self.assertFalse(r.is_meta)


if __name__ == '__main__':
    run_tests()
