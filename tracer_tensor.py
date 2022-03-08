import torch
from torch import Tensor
from torch.fx import Tracer, GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests

from utils import no_dispatch


class TracerTensor(Tensor):
    __slots__ = ['proxy']

    @staticmethod
    def __new__(cls, elem, proxy):
        # Unlike other tensor subclasses in the zoo, TracerTensor detaches
        # autograd upon creation.  (Is this right?)
        return Tensor._make_subclass(cls, elem, elem.requires_grad)

    def __init__(self, elem, proxy):
        # elem does not need to be recorded, because TracerTensor *is a* elem
        self.proxy = proxy
        # Since the proxy is associated with a concrete Tensor object, we also
        # know exactly what its tensor metadata should be, so populate it
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(self)

    def __repr__(self):
        with no_dispatch():
            return f"TracerTensor({repr(self)})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_proxy(t):
            if isinstance(t, cls):
                return t.proxy
            else:
                return t

        def wrap(t, p):
            # Some ops (like native_batch_norm_backward) return undefined
            # tensor that get converted into None in Python.  This papers
            # over this problem.
            # TODO: fix this inside core so that None returns from
            # __torch_dispatch__ turn into undefined tensors
            if t is None:
                return torch.empty(())
            elif isinstance(t, Tensor):
                return TracerTensor(t, p)
            else:
                assert t == p
                return t

        with no_dispatch():
            r = func(*args, **kwargs)

        r_proxy = func(
            *tree_map(unwrap_proxy, args),
            **tree_map(unwrap_proxy, kwargs))

        # NB: we cannot zip r and r_proxy, or rely on r_proxy knowing its
        # structure, because r_proxy as implemented in FX typically is a proxy
        # that will generate IR for accessing subfields as you invoke.  So
        # r has to "drive" the deconstruction.
        # NB: this assumes there aren't any recursive return structs, which
        # is generally a safe bet in the current codegen
        if isinstance(r, list):
            return [wrap(t, r_proxy[i]) for i, t in enumerate(r)]
        elif isinstance(r, tuple):
            return tuple(wrap(t, r_proxy[i]) for i, t in enumerate(r))
        else:
            return wrap(r, r_proxy)


class TracerTensorTest(TestCase):
    def test_basic(self):
        t = Tracer()


if __name__ == '__main__':
    run_tests()
