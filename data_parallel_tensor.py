import functools
import torch

import torch
from torch._C import device
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import push_torch_dispatch_mode, TorchDispatchMode
aten = torch.ops.aten

class DataParallelTensor(torch.Tensor):
    elem: List[torch.Tensor]
    device_ids: List[int]
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, func=None):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
        )
        r.elem = []
        pos = 0
        for device_id in r.device_ids:
            if(elem.device == device(device_id)):
                r.elem[pos] = elem
            else:
                r.elem[pos] = elem.to(device = device_id)
            pos += 1

        return r

    def __repr__(self):
        if self.grad_fn:
            return f"DataParallelTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"DataParallelTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, DataParallelTensor) else e

        outs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if e is None:
                return torch.empty(())
            return DataParallelTensor(e, func) if isinstance(e, torch.Tensor) else e

        outs = tree_map(wrap, outs)
        return outs
