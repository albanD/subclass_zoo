import torch
from torch import Tensor
from torch.fx import Tracer
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import tree_map

from utils import no_dispatch, tree_map2
from typing import Any


class TracerTensor(Tensor):
    __slots__ = ['proxy']

    @staticmethod
    def __new__(cls, elem, proxy):
        # See Note [Passing requires_grad=true tensors to subclasses]
        assert not elem.requires_grad or not torch.is_grad_enabled()
        return Tensor._make_subclass(cls, elem)

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
            # TODO: fix this
            if t is None:
                return torch.empty(())
            elif isinstance(t, Tensor):
                return TracerTensor(t, p)
            else:
                assert t == p
                return t

        with no_dispatch():
            return tree_map2(
                wrap,
                func(*args, **kwargs),
                func(
                    *tree_map(unwrap_proxy, args),
                    **tree_map(unwrap_proxy, kwargs))
            )
