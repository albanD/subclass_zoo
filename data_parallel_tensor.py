import torch
from torch._C import device
from torch.functional import Tensor
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
    _get_devices_properties,
)

aten = torch.ops.aten


class DataParallelTensor(torch.Tensor):
    elem: List[torch.Tensor]
    device_ids: List[int] = _get_all_device_indices()
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, func=None, replicate=False):
        meta_t = elem if replicate else elem[0]
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            meta_t.size(),
            strides=meta_t.stride(),
            storage_offset=meta_t.storage_offset(),
            device=meta_t.device, #This is the device of of either input tensor or first tensor of a list
            dtype=meta_t.dtype, 
            layout=meta_t.layout,
            requires_grad=meta_t.requires_grad,
        )
        if replicate:
            r.elem = []
            with torch.no_grad():
                for device_id in r.device_ids:
                    t: torch.Tensor = elem.to(device=device_id)
                    t.requires_grad = elem.requires_grad
                    r.elem.append(t)
                    t = None
        else:
            assert isinstance(elem, list)
            pos = 0
            with torch.no_grad():
                for t, d_id in zip(elem, r.device_ids):
                    if t.device != device(d_id):
                        elem[pos] = t.to(device=d_id)
                        elem[pos].requires_grad = t.requires_grad
                    pos += 1
            r.elem = elem

        return r

    def __repr__(self):
        if self.grad_fn:
            return f"DataParallelTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"DataParallelTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):

        def wrap(e):
            if (isinstance(e, DataParallelTensor)):
                return e
            elif(isinstance(e, torch.Tensor)):
                return DataParallelTensor(e, func, True)
            else:
                return e

        args = tree_map(wrap, args)
        kwargs = tree_map(wrap, kwargs)

        def unwrap_with_position(pos):
            def get_element(e):
                return e.elem[pos] if isinstance(e, DataParallelTensor) else e

            return get_element

        outs = []
        for pos in range(len(cls.device_ids)):
            # import pdb
            # if(func == aten.mul.Tensor):
            #     pdb.set_trace()
            outs.append(
                func(
                    *tree_map(unwrap_with_position(pos), args),
                    **tree_map(unwrap_with_position(pos), kwargs),
                )
            )
        # outs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def get_element_type(lis):
            assert isinstance(lis, list)
            return type(lis[0])

        def out_wrap(e, func):
            if e is None:
                return torch.empty(())
            elem_type = get_element_type(e)

            if elem_type == torch.Tensor:
                return DataParallelTensor(outs, func)
            elif elem_type == list:
                return list(DataParallelTensor(list(t), func) for t in zip(*e))
            elif elem_type == tuple:
                return tuple(DataParallelTensor(list(t), func) for t in zip(*e))

        # outs = tree_map(wrap, outs)
        outs = out_wrap(outs, func)
        return outs


print(_get_all_device_indices())
test_tensor = torch.randn(5, device="cuda", requires_grad=True)
dp_tensor = DataParallelTensor(test_tensor, None, True)
res_tensor = dp_tensor.cos().cos().sum()
print(res_tensor)
test_tensor.to(device="cuda")
res_tensor.backward()
print(dp_tensor.grad)
