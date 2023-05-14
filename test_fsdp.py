import torch
from torch.utils import _pytree as pytree
from torch.autograd import Function
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
from torch.utils._mode_utils import no_dispatch
import torchviz


### FSDP PART
class SlicedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, local_tensor, base_tensor):
        kwargs = {}
        kwargs["device"] = local_tensor.device
        kwargs["dtype"] = local_tensor.dtype
        kwargs["layout"] = local_tensor.layout
        self = torch.Tensor._make_wrapper_subclass(cls, local_tensor.size(), **kwargs)
        # self = torch.Tensor._make_subclass(cls, local_tensor)
        self._local_tensor = local_tensor
        self._base_tensor = base_tensor
        self._all_slices = None
        return self

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        # This is (1) sad that we have to special case (2) Parameter printing is so broken
        if isinstance(self, nn.Parameter):
            prefix = 'Parameter containing:\n'
        else:
            prefix = ''
        return prefix + f"Sliced({self._local_tensor}, full_backing_size={self._base_tensor.size()})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        if func is torch.ops.aten.detach.default:
            return args[0]
        if func is torch.ops.aten.set_.source_Tensor:
            inp, new = args
            if inp is new:
                return inp
            else:
                raise NotImplementedError("NYI")
        else:
            args = pytree.tree_map_only(SlicedTensor, lambda x: x._local_tensor, args)
            return func(*args, **kwargs)

    @staticmethod
    def from_full_tensor(full_w):
        slices = full_w.unbind()
        all_slices = tuple(SlicedTensor(s, full_w) for s in slices)
        for s in all_slices:
            s._all_slices = all_slices
        return all_slices

class UnsliceReduceFn(Function):
    @staticmethod
    def forward(ctx, *all_w):
        assert all(w._base_tensor is all_w[0]._base_tensor for w in all_w)
        return all_w[0]._base_tensor.detach()

    @staticmethod
    def backward(ctx, gO):
        # gO = td.all_reduce(gO)
        return gO.unbind()

class UnsliceReduceGrad(nn.Module):
    def __init__(self, unslice_state):
        super().__init__()
        self.unslice_state = unslice_state

    def forward(self, w):
        assert isinstance(w, SlicedTensor)
        if len(self.unslice_state) > 0:
            return self.unslice_state[0]
        else:
            new_full_w = UnsliceReduceFn.apply(*w._all_slices)
            # Clear the state after the backward call
            def clean_state_hook(_):
                self.unslice_state.pop()
            new_full_w.register_hook(clean_state_hook)
            self.unslice_state.append(new_full_w)
            return new_full_w

class Slice(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, w):
        return w[self.idx]

def make_fsdp_(mod):
    # Concat all weights together
    full_w = []
    for l in mod:
        full_w.append(l.weight)
    with torch.no_grad():
        full_w = torch.stack(full_w)
        new_w = SlicedTensor.from_full_tensor(full_w)

    # Update the weights to the new view
    unslice_state = []
    for i, l in enumerate(mod):
        l.weight = nn.Parameter(new_w[i])

        register_parametrization(l, "weight", UnsliceReduceGrad(unslice_state), unsafe=True)
        register_parametrization(l, "weight", Slice(i), unsafe=True)

### Quantization part
class QuantizedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, dtype, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=dtype)

    def __init__(self, data, dtype, requires_grad=False):
        self._data = data

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        # This is (1) sad that we have to special case (2) Parameter printing is so broken
        if isinstance(self, nn.Parameter):
            prefix = 'Parameter containing:\n'
        else:
            prefix = ''
        return prefix + f"QuantizedTensor({self._data}, public_dtype={self.dtype})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        args, kwargs = pytree.tree_map_only(QuantizedTensor, lambda x: x._data, (args, kwargs or {}))
        # Always downcast everything to bfloat16 for simplicity
        if pytree.tree_any_only(torch.Tensor, lambda x: x.dtype is torch.bfloat16, args):
            args, kwargs = pytree.tree_map_only(torch.Tensor, lambda x: x.to(torch.bfloat16) if x.dtype is torch.float32 else x, (args, kwargs))
        raw_out = func(*args, **kwargs)
        out = pytree.tree_map_only(torch.Tensor, lambda x: QuantizedTensor(x, torch.float32), raw_out)
        return out

def quantize_params_(mod):
    for l in mod:
        l.weight = nn.Parameter(QuantizedTensor(l.weight.to(torch.bfloat16), torch.float32))


### TESTING

# Quant + FSDP
mod = nn.Sequential(
    nn.Linear(2, 2),
    nn.Linear(2, 2),
    nn.Linear(2, 2),
)

print(mod)
print(mod[0].weight)
print(list(mod[0].parameters()))

quantize_params_(mod)

print(mod)
print(mod[0].weight)
print(list(mod[0].parameters()))

make_fsdp_(mod)

print(mod)
print(mod[0].weight)
print(list(mod[0].parameters()))

