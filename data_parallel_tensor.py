import torch
from torch._C import device, NoneType
from torch.cuda import comm
from torch.functional import Tensor
import torch.nn as nn
import pdb
from torch.return_types import _fake_quantize_per_tensor_affine_cachemask_tensor_qparams
from torch.utils._pytree import tree_map, tree_flatten
from typing import Iterable, List, Any, Optional, Tuple, Iterable, Union
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import push_torch_dispatch_mode, TorchDispatchMode
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
    _get_devices_properties,
)
from enum import Enum, auto



aten = torch.ops.aten

class DPTensorType(Enum):
    replicated = auto() #This tensor will be replicated across all the devices
    sharded = auto() #This tensor will be sharded along the first/batch dimension across the devices, NOTE: only equal chunk sizes are supported
    distributed = auto() # This is a list of tensors, each of which rests on different devices


class DataParallelTensor(torch.Tensor):
    # This class is a tensor subclass that stores a list of tensors with the aim
    # DataParallelTensors(DPT) are categorized in three ways
    # 1) replicated: When a single tensor is supplied, it is replicated across
    #   all the devices by using broadcast
    # 2) distributed: DPT can also be initialized by supplying a list/tuple of tensors
    #   if the elements rest on different devices, they will just be wrapped in DPT
    #   else the elements are scattered to different devices
    # 3) sharded: This type of DPT tensor is created by sharding the input tensor across
    #   a specified sharding dimension (default: 0). Currently only equal chunk sizes are supported.

    elem: List[torch.Tensor]
    device_ids: List[int] = _get_all_device_indices()
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem: Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor]], func:Optional[Any] = None, dpt_type:DPTensorType = DPTensorType.replicated, shard_dim:Optional[int] = 0):

        if dpt_type == DPTensorType.replicated:
            assert isinstance(elem, torch.Tensor)
            with torch.no_grad():
                dpt:List[torch.Tensor] = comm.broadcast(elem, devices=cls.device_ids)
                for t in dpt:
                    t.requires_grad = elem.requires_grad

        elif dpt_type == DPTensorType.distributed:
            #This can work on list or tuple of tensors
            assert (isinstance(elem, list) or isinstance(elem, tuple))

            requires_scatter:bool = False
            with torch.no_grad():
                for t, d_id in zip(elem, cls.device_ids):
                    if t.device != device(d_id):
                        requires_scatter = True
                        break

                if(requires_scatter):
                    stacked_t:torch.Tensor = torch.stack(elem, dim =0)
                    scattered_t: Tuple[torch.Tensor] = comm.scatter(stacked_t, devices = cls.device_ids, dim = 0)
                    dpt:List[torch.Tensor] = [torch.squeeze(t, dim = 0) for t in scattered_t]
                    for t, e in zip(dpt, elem):
                        t.requires_grad = e.requires_grad
                else:
                    dpt:List[torch.Tensor] = elem
        elif dpt_type == DPTensorType.sharded:
            assert(isinstance(elem, torch.Tensor))

            with torch.no_grad():
                scattered_t:Tuple[torch.Tensor] = comm.scatter(elem, devices = cls.device_ids, dim = shard_dim)
                dpt:List[torch.Tensor] = list(scattered_t)
                for t in dpt:
                    t.requires_grad = elem.requires_grad    

        meta_t:torch.Tensor = dpt[0]
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
        r.elem = dpt

        return r

    def __repr__(self):
        if self.grad_fn:
            return f"DataParallelTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"DataParallelTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(func)

        def wrap(e):
            if (isinstance(e, DataParallelTensor)):
                return e
            elif(isinstance(e, torch.Tensor)):
                return DataParallelTensor(e, func, DPTensorType.replicated)
            else:
                return e
        # All the args and kwargs are checked and any leaf tensors are wrapped as replicated DPTs
        args = tree_map(wrap, args)
        kwargs = tree_map(wrap, kwargs)

        def unwrap_with_position(pos):
            def get_element(e):
                return e.elem[pos] if isinstance(e, DataParallelTensor) else e

            return get_element
        # Call the function for each of the DPT elements by unwarpping them and corresponding args and kwargs into tensors
        outs = []

        for pos in range(len(cls.device_ids)):
            # import pdb
            if (func == aten.convolution_backward.default):
                pdb.set_trace()
            outs.append(
                func(
                    *tree_map(unwrap_with_position(pos), args),
                    **tree_map(unwrap_with_position(pos), kwargs),
                )
            )

        def get_element_type(lis):
            assert isinstance(lis, list)
            return type(lis[0])

        # The ouput will always be a list
        # The list can contain tensors, list of tensors or tuples of tensors
        # In case of tensors we just wrap them in DPT
        # In case of list/tuple of tensors, the corresponding elemsnts across list/tuple are warpped
        #  into a DPT and a list/tuple is returned respectively

        def out_wrap(e, func):
            elem_type = get_element_type(e)
            if elem_type is NoneType:
                return torch.empty(()) #NOTE: Maybe should return a list of torch.empty()            
            if elem_type == torch.Tensor:
                return DataParallelTensor(outs, func, DPTensorType.distributed)
            elif elem_type == list:
                return list(DataParallelTensor(list(t), func, DPTensorType.distributed) for t in zip(*e))
            elif elem_type == tuple:
                return tuple(DataParallelTensor(list(t), func, DPTensorType.distributed) for t in zip(*e))

        outs = out_wrap(outs, func)
        return outs


print(_get_all_device_indices())
test_tensor = torch.randn(32,3, 224, 224, device="cuda")
dp_tensor = DataParallelTensor(test_tensor, None, DPTensorType.sharded)
# res_tensor = dp_tensor.cos().cos().sum()
# print(res_tensor)
# res_tensor.backward()
# print(dp_tensor.grad)

import torchvision.models as models


model = models.resnet18().cuda()

out = model(dp_tensor)
loss = out.sum()
print(type(loss))
print(loss.size())
loss.backward()
print(type(dp_tensor.grad))
print(dp_tensor.grad.size())
