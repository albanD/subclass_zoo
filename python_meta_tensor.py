import itertools

import torch

import torch.nn
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._python_dispatch import enable_python_mode
from torch.utils._pytree import tree_flatten, tree_map

aten = torch.ops.aten
aten.__origin__ = None

"""
Meta tensors give you the ability to run PyTorch code without having to
actually do the compute, which is useful if you only need to figure out what
output sizes might be or need to trace a program. However, meta device support
in PyTorch is somewhat spotty, as we have mostly gotten meta tensor support by
porting kernels to structured kernels, which is a relatively time consuming
process (although it ensures that our meta implementations are 100% correct,
as they are derived from a single source of truth).

This idea for solving this problem originally comes from Can Balioglu at
https://github.com/pytorch/pytorch/pull/66317/
With tensor subclasses, we can create a subclass of meta tensor,
PythonMetaTensor, which manually adds support for missing meta device
implementations. Indeed, we can even implement this as a mode, so that when a
context manager is active, we interpose on all operations on meta tensors and
override the behavior of some operations with our own implementations.

I found it very pleasant and quick writing Python implementations for the meta
functions; feedback was instantaneous without any C++ compilation cycle. These
implementations could then be ported to C++ (short term), or removed entirely
when the kernels in question turned structured (long term).

Note that https://github.com/pytorch/pytorch/pull/62660 would have also had a
similar effect, but at time of writing it is not landed in core, so I shipped
the version using subclasses/modes instead.
"""


# TODO: duplicated from utils.py
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
        r.append(defaults_tail[i - n + len(defaults_tail)])
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
        if not any(
            isinstance(t, torch.Tensor) and t.is_meta
            for t in itertools.chain(flat_args, flat_kwargs)
        ) and kwargs.get("device", None) != torch.device("meta"):
            return super().__torch_dispatch__(func, types, args, kwargs)

        if func == aten._embedding_bag.default:
            # Defaults can be determined by reading native_functions.yaml
            # We will soon make these available directly from the torch.ops
            # API, waiting on https://github.com/pytorch/pytorch/pull/72673
            args = fill_defaults(args, 9, [False, 0, False, None, False, -1])
            (
                weight,
                indices,
                offsets,
                scale_grad_by_freq,
                mode,
                sparse,
                per_sample_weights,
                include_last_offset,
                padding_idx,
            ) = args
            assert not kwargs
            # I determined the meaning of the outputs and sizes by reading
            # over the kernel in aten/src/ATen/native/EmbeddingBag.cpp
            output = weight.new_empty(
                offsets.size(0) - 1 if include_last_offset else offsets.size(0),
                weight.size(1),
            )
            MODE_SUM, MODE_MEAN, MODE_MAX = range(3)
            if mode == MODE_MEAN or mode == MODE_MAX:
                offset2bag = offsets.new_empty(indices.size(0))
            else:
                offset2bag = offsets.new_empty(0)
            bag_size = offsets.new_empty(offsets.size())
            max_indices = offsets.new_empty(bag_size.size())
            return output, offset2bag, bag_size, max_indices
        elif func == aten.index_select.default:
            # TODO: when I didn't have embedding implemented, it reported that
            # index_select wasn't implemented, but it didn't actually help to
            # implement this (because once we go to the
            # CompositeExplicitAutograd, Python key is disabled and we won't
            # come back here).  Oof.
            self, dim, index = args
            assert not kwargs
            result_size = list(self.size())
            if self.dim() > 0:
                result_size[dim] = index.numel()
            return self.new_empty(result_size)
        elif func == aten.embedding.default:
            args = fill_defaults(args, 5, [-1, False, False])
            weight, indices, padding_idx, scale_grad_by_freq, sparse = args
            assert not kwargs
            assert weight.dim() == 2
            assert indices.dtype in [torch.long, torch.int]
            if indices.dim() == 1:
                return weight.index_select(0, indices)
            size = list(indices.size())
            size.extend(weight.size()[1:])
            return weight.index_select(0, indices.reshape(-1)).view(size)
        elif func == aten._linalg_qr_helper.default:
            input, mode = args
            assert not kwargs
            if mode == "reduced":
                compute_q = True
                reduced_mode = True
            elif mode == "complete":
                compute_q = True
                reduced_mode = False
            elif mode == "r":
                compute_q = False
                reduced_mode = True
            else:
                raise RuntimeError(f"qr received unrecognized mode {mode}")
            m = input.size(-2)
            n = input.size(-1)
            mn = min(m, n)
            if compute_q:
                Qt_shape = list(input.size())
                Qt_shape[-2] = mn if reduced_mode else m
                Qt_shape[-1] = m
                Q = input.new_empty(Qt_shape)
                Q.transpose_(-2, -1)
            else:
                Q = input.new_empty(0)
            Rt_shape = list(input.size())
            Rt_shape[-2] = n
            Rt_shape[-1] = mn if reduced_mode or not compute_q else m
            R = input.new_empty(Rt_shape)
            R.transpose_(-2, -1)
            return (Q, R)
        elif func == aten.linalg_qr.default:
            self, mode = fill_defaults(args, 2, ["reduced"])
            assert not kwargs
            assert self.dim() >= 2
            return aten._linalg_qr_helper(self, mode)
        elif func == aten.inverse.default:
            (self,) = args
            assert not kwargs
            if self.numel() == 0:
                return self.new_empty(self.size())
            inverse = self.new_empty(self.size())
            inverse.transpose_(-2, -1)
            return inverse
        elif func == aten.randperm.default:
            (n,) = args
            # intentionally no assert not kwargs
            # TODO: dtype shows up as int which is bad; should convert
            # this as torch.dtype when it gets here.  Fortunately
            # forwarding to torch.ops the integer will be understood.
            return torch.ops.aten.empty((n,), **kwargs)
        elif func == aten.max.default:
            (self,) = args
            assert not kwargs
            return self.new_empty(())
        elif func == aten.sort.default:
            self, dim, descending = fill_defaults(args, 3, [-1, False])
            assert not kwargs
            return self.new_empty(self.size()), self.new_empty(
                self.size(), dtype=torch.long
            )
        elif func == aten.repeat_interleave.Tensor:
            (repeats,) = args
            output_size = kwargs.pop("output_size", None)
            assert not kwargs
            if output_size is None:
                raise RuntimeError(
                    "cannot repeat_interleave a meta tensor without output_size"
                )
            return repeats.new_empty(output_size)
        elif func == aten._det_lu_based_helper.default:
            (self,) = args
            assert not kwargs
            pivs_size = list(self.size()[:-2])
            pivs_size.append(min(self.size(-1), self.size(-2)))
            return (
                self.new_empty(()),
                self.new_empty(self.size()),
                self.new_empty(pivs_size, dtype=torch.int),
            )
        elif func == aten.abs_.default:
            (self,) = args
            # TODO: assert self not complex
            assert not kwargs
            return self
        elif func == aten.abs.default:
            (self,) = args
            assert not kwargs
            if self.is_complex():
                from_complex = {torch.cfloat: torch.float, torch.cdouble: torch.double}
                float_type = from_complex[self.dtype]
                self.new_empty(self.size(), dtype=float_type)
            else:
                return self.new_empty(self.size())
        elif func == aten.complex.default:
            real, imag = args
            assert real.dtype == imag.dtype
            assert not kwargs
            assert real.size() == imag.size()
            to_complex = {torch.float: torch.cfloat, torch.double: torch.cdouble}
            return real.new_empty(real.size(), dtype=to_complex[real.dtype])
        elif func == aten.eye.default:
            (n,) = args
            # intentionally no assert not kwargs
            return torch.ops.aten.empty((n, n), **kwargs)
        elif func == aten.linalg_cholesky_ex.default:
            (input,) = args
            upper = kwargs.pop("upper", False)
            check_errors = kwargs.pop("check_errors", False)
            assert not kwargs
            info_output_dtype = torch.int
            # TODO: check linalg compatible dtype
            # linalg_cholesky_out_info
            assert input.dim() >= 2
            assert input.size(-1) == input.size(-2)
            L_sizes = list(input.size())
            L_sizes[-1], L_sizes[-2] = L_sizes[-2], L_sizes[-1]
            L = input.new_empty(L_sizes)
            L.transpose_(-2, -1)
            info_sizes = input.size()[:-2]
            info = input.new_empty(info_sizes, dtype=torch.int)
            return L, info
        elif func == aten._linalg_check_errors.default:
            return
        elif func == aten.lu_unpack.default:
            args = fill_defaults(args, 4, [True, True])
            LU_data, LU_pivots, unpack_data, unpack_pivots = args
            L = None
            U = None
            m = LU_data.size(-2)
            n = LU_data.size(-1)
            k = min(m, n)
            if unpack_data:
                U = LU_data.tril()
                if m != k:
                    U = U.narrow(-2, 0, k)
                L = LU_data.triu()
                if k != n:
                    L = L.narrow(-1, 0, k)
            if not unpack_pivots:
                return None, L, U
            unpacked_pivots_sizes = list(LU_pivots.size())
            unpacked_pivots_sizes[-1] = m
            unpacked_pivots_sizes.append(m)
            # TODO: layout is not done correctly
            permutation_matrix = LU_data.new_empty(unpacked_pivots_sizes)
            return permutation_matrix, L, U
        # add your other patches here

        try:
            return super().__torch_dispatch__(func, types, args, kwargs)
        except NotImplementedError:
            # TODO: aten._local_scalar_dense.default is special, you can't
            # implement it, add a special case for it
            raise NotImplementedError(f"no meta implementation for {func}")


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


class PythonMetaTensorTest(TestCase):
    def test_basic(self):
        x = PythonMetaTensor(torch.empty(2, 2, device="meta"))
        y = x + x
        self.assertEqual(y.shape, (2, 2))

    def test_embedding_bag(self):
        embedding_sum = torch.nn.EmbeddingBag(10, 3, mode="sum", device="meta")
        input = torch.empty(8, dtype=torch.long, device="meta")
        offsets = torch.empty(2, dtype=torch.long, device="meta")
        self.assertRaises(NotImplementedError, lambda: embedding_sum(input, offsets))
        r = embedding_sum(PythonMetaTensor(input), PythonMetaTensor(offsets))
        self.assertEqual(r, torch.empty((2, 3), dtype=torch.float, device="meta"))

    def test_embedding_via_mode(self):
        with enable_python_mode(PythonMetaTensorMode):
            embedding = torch.nn.Embedding(10, 3, device="meta")
            input = torch.empty((2, 4), dtype=torch.long, device="meta")
            r = embedding(input)
            self.assertEqual(
                r, torch.empty((2, 4, 3), dtype=torch.float, device="meta")
            )

    def test_embedding_bag_via_mode(self):
        with enable_python_mode(PythonMetaTensorMode):
            embedding_sum = torch.nn.EmbeddingBag(10, 3, mode="sum", device="meta")
            input = torch.empty(8, dtype=torch.long, device="meta")
            offsets = torch.empty(2, dtype=torch.long, device="meta")
            r = embedding_sum(input, offsets)
            self.assertEqual(r, torch.empty((2, 3), dtype=torch.float, device="meta"))

            # Make sure we don't interpose on non-meta computation
            embedding_sum = torch.nn.EmbeddingBag(10, 3, mode="sum")
            input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
            offsets = torch.tensor([0, 4], dtype=torch.long)
            r = embedding_sum(input, offsets)
            self.assertFalse(r.is_meta)


if __name__ == "__main__":
    run_tests()
