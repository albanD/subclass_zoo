import functools
import torch

import torch
from torch._C import device
import torch.nn as nn
from torch.return_types import _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import push_torch_dispatch_mode, TorchDispatchMode
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties
)
aten = torch.ops.aten

class DataParallelTensor(torch.Tensor):
    elem: List[torch.Tensor]
    device_ids: List[int] = _get_all_device_indices()
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, func=None, replicate=False):
        
        if(replicate):
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
            for device_id in r.device_ids:
                r.elem.append(elem.to(device = device_id))
        else:
            assert (isinstance(elem, list))
            r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem[0].size(),
            strides=elem[0].stride(),
            storage_offset=elem[0].storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem[0].dtype,
            layout=elem[0].layout,
            requires_grad=elem[0].requires_grad,
            )
            r.elem = elem

        return r

    def __repr__(self):
        if self.grad_fn:
            return f"DataParallelTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"DataParallelTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def unwrap_with_position(pos):
            def get_element(e):
                return e.elem[pos] if isinstance(e, DataParallelTensor) else e
            return get_element

        outs = []
        for pos in range(len(cls.device_ids)):
            outs.append(func(*tree_map(unwrap_with_position(pos), args), **tree_map(unwrap_with_position(pos), kwargs)))
        # outs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if e is None:
                return torch.empty(())
            return DataParallelTensor(e, func) if isinstance(e, torch.Tensor) else e

        # outs = tree_map(wrap, outs)
        outs = DataParallelTensor(outs, func)
        return outs

print(_get_all_device_indices())
test_tensor = torch.randn(5, device = 'cuda')
dp_tensor = DataParallelTensor(test_tensor, None ,True)
res_tensor = dp_tensor.cos()
print(res_tensor)
    
