import torch
from torch._C import device, NoneType
import torch.nn.functional as F
from torch.cuda import comm
from torch.functional import Tensor
import torch.nn as nn
import pdb
import weakref
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
import torchvision.models as models
from functools import partial
_ = torch.manual_seed(0)

torch.__future__.set_overwrite_module_params_on_conversion(True)

aten = torch.ops.aten

class DPTensorType(Enum):
    replicated = auto() #This tensor will be replicated across all the devices
    distributed_batch = auto() #This tensor will be sharded along the first/batch dimension across the devices, NOTE: only equal chunk sizes are supported
    distributed = auto() # This is a list of tensors, each of which rests on different devices


class DataParallelTensor(torch.Tensor):
    # This class is a tensor subclass that stores a list of tensors with the aim
    # DataParallelTensors(DPT) are categorized in three ways
    # 1) replicated: When a single tensor is supplied, it is replicated across
    #   all the devices by using broadcast
    # 2) distributed: DPT can also be initialized by supplying a list/tuple of tensors
    #   if the elements rest on different devices, they will just be wrapped in DPT
    #   else the elements are scattered to different devices
    # 3) distributed batch: This type of DPT tensor is created by sharding the input tensor across
    #   a specified sharding dimension (default: 0). Currently only equal chunk sizes are supported.

    elem: List[torch.Tensor]
    device_ids: List[int] = _get_all_device_indices()
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem: Union[torch.Tensor,List[torch.Tensor],Tuple[torch.Tensor]], func:Optional[Any] = None, dpt_type:DPTensorType = DPTensorType.replicated, batch_dim:Optional[int] = 0):


        if dpt_type == DPTensorType.replicated:
            assert isinstance(elem, torch.Tensor)
            if(elem.device == device('meta')):
                return elem
            with torch.no_grad():
                dpt:List[torch.Tensor] = comm.broadcast(elem, devices=cls.device_ids)

        elif dpt_type == DPTensorType.distributed:
            #This can work on list or tuple of tensors
            # breakpoint()
            assert (isinstance(elem, list) or isinstance(elem, tuple))
            check_none = [True if e is None else False for e in elem]
            if any(check_none):
                return None
            check_not_tensor = [True if not isinstance(e, torch.Tensor) else False for e in elem]
            if any(check_not_tensor):
                #NOTE: Need to define behaviour when an operation returns a tuple/list of vlaues that are not tensors
                raise RuntimeWarning("Expected Tensor type in DataParallelTensor class Constructor")
                return elem[0]
            requires_scatter:bool = False
            with torch.no_grad():
                for t, d_id in zip(elem, cls.device_ids):
                    if(t.device == device('meta')):
                        return elem[0]
                    if t.device != device(d_id):
                        requires_scatter = True
                        break

                if(requires_scatter):
                    stacked_t:torch.Tensor = torch.stack(elem, dim =0)
                    scattered_t: Tuple[torch.Tensor] = comm.scatter(stacked_t, devices = cls.device_ids, dim = 0)
                    dpt:List[torch.Tensor] = [torch.squeeze(t, dim = 0) for t in scattered_t]
                else:
                    dpt:List[torch.Tensor] = elem
        elif dpt_type == DPTensorType.distributed_batch:
            assert(isinstance(elem, torch.Tensor))

            with torch.no_grad():
                scattered_t:Tuple[torch.Tensor] = comm.scatter(elem, devices = cls.device_ids, dim = batch_dim)
                dpt:List[torch.Tensor] = list(scattered_t)  

        meta_t:torch.Tensor = elem if dpt_type in (DPTensorType.replicated, DPTensorType.distributed_batch) else elem[0]
        #NOTE: Check what needs to be done for distributes_batch case
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
                return None #NOTE: Maybe should return a list of torch.empty()            
            if elem_type == torch.Tensor:
                return DataParallelTensor(outs, func, DPTensorType.distributed)
            elif elem_type == list:
                return list(DataParallelTensor(list(t), func, DPTensorType.distributed) for t in zip(*e))
            elif elem_type == tuple:
                return tuple(DataParallelTensor(list(t), func, DPTensorType.distributed) for t in zip(*e))
            elif elem_type == bool:
                return all(e)
            else:
                # NOTE: Think about handling this
                print("Warning...")
                return e[0]

        outs = out_wrap(outs, func)
        return outs
    
    def all_reduce_grad(self, r_device:Optional[int]= torch.cuda.current_device()):
        with torch.no_grad():
            reduced_tensor: torch.Tensor = comm.reduce_add(self.elem,r_device)
            b_tensor:List[torch.Tensor] = comm.broadcast(reduced_tensor, out=self.elem)
            self.elem = b_tensor
        return reduced_tensor

def make_data_parallel_module(mod: torch.nn.Module):
    # This function converts the parameters of a model to DataParallelTensors
    def wrapper(t):
        if(isinstance(t, torch.nn.Parameter)):
            return  DataParallelTensor(t.data, None, DPTensorType.replicated)
    mod._apply(wrapper)

print("Devices: ", _get_all_device_indices())
D = 16
dpt_x: torch.Tensor = torch.randn(D, device='cuda', requires_grad=True)
dpt_x = DataParallelTensor(dpt_x, None, DPTensorType.replicated)
def predict(weight, bias, x):
    return F.linear(x, weight, bias).tanh()

weight = torch.randn(D, D, device = 'cuda')
bias = torch.randn(D, device = 'cuda')


unit_vectors = torch.eye(D).cuda()



# from functorch import vmap, vjp

# _, vjp_fn = vjp(partial(predict, weight, bias), dpt_x)

# ft_jacobian, = vmap(vjp_fn)(unit_vectors)
# print(type(ft_jacobian))

from functorch import jacrev, hessian, jacfwd

# ft_jacobian = jacfwd(predict, argnums=2)(weight, bias, dpt_x)
# # ft_jacobian2 = jacfwd(ft_jacobian, argnums=2)(weight, bias, dpt_x)
# ft_jacobian2 = jacrev(predict, argnums=2)(weight, bias, dpt_x)
# print(torch.allclose(ft_jacobian.elem[0], ft_jacobian2.elem[0]))

# # print(ft_jacobian)

hess_api = hessian(predict, argnums=2)(weight, bias, dpt_x)
hess_fwdfwd = jacfwd(jacfwd(predict, argnums=2), argnums=2)(weight, bias, dpt_x)
print(torch.allclose(hess_api, hess_fwdfwd))

# def compute_jac(xp):
#     jacobian_rows = [torch.autograd.grad(predict(weight, bias, xp), xp, vec)[0]
#                      for vec in unit_vectors]
#     return torch.stack(jacobian_rows)

# jacobian = compute_jac(dpt_x)

# print(jacobian.shape)
# print(jacobian[0])

# Example with a model
# dp_tensor = DataParallelTensor(test_tensor, None, DPTensorType.distributed_batch)
# model = models.resnet18().cuda()
# make_data_parallel_module(model)
# out = model(test_tensor)
# loss = out.sum()
# print(type(loss))
# print(loss.size())
# loss.backward()

# for p in model.parameters():
#     print(type(p.data))
#     print(type(p.grad))
#     print(type(p.grad.elem))
#     p.grad.all_reduce_grad(device('cuda'))
#     p = p - 0.5 * p.grad

# Non Model Example
# res_tensor = dp_tensor.cos().cos().sum()
# print(res_tensor)
# res_tensor.backward()
# print(dp_tensor.grad)
