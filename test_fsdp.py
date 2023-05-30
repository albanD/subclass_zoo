import torch
from torch.utils import _pytree as pytree
from torch.autograd import Function
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

### FSDP PART
class BucketedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, view_tensor, base_tensor, idx):
        kwargs = {}
        kwargs["device"] = view_tensor.device
        kwargs["dtype"] = view_tensor.dtype
        kwargs["layout"] = view_tensor.layout
        self = torch.Tensor._make_wrapper_subclass(cls, view_tensor.size(), **kwargs)
        self._view_tensor = view_tensor
        self._base_tensor = base_tensor
        self._idx = idx
        self._all_views = None
        return self

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):
        # This is (1) sad that we have to special case as (2) Parameter printing is so broken
        if isinstance(self, nn.Parameter):
            prefix = 'Parameter containing:\n'
        else:
            prefix = ''
        return prefix + f"Sliced({self._view_tensor}, full_backing_size={self._base_tensor.size()})"

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
            args = pytree.tree_map_only(BucketedTensor, lambda x: x._view_tensor, args)
            return func(*args, **kwargs)

    @staticmethod
    def from_full_tensor(full_w):
        assert not torch.is_grad_enabled()
        slices = full_w.unbind(dim=-1)
        all_slices = tuple(BucketedTensor(s, full_w, i) for i, s in enumerate(slices))
        for s in all_slices:
            s._all_views = all_slices
        return all_slices

class UnsliceReduceFn(Function):
    @staticmethod
    def forward(ctx, rank, *all_w):
        assert all(w._base_tensor is all_w[0]._base_tensor for w in all_w)
        full_local = all_w[0]._base_tensor.detach()
        # Assume everyone is involved
        ctx.idxs = list(range(WORLD_SIZE))
        ctx.rank = rank
        out = torch.distributed._functional_collectives.all_gather_tensor(full_local, 0, ctx.idxs,"huh")
        return out

    @staticmethod
    def backward(ctx, gO):
        # gO = td.all_reduce(gO)
        # Should be reduce_scatter but that doesn't work on gloo
        gO = torch.distributed._functional_collectives.all_reduce(gO, "sum", ctx.idxs)
        gO = gO.chunk(WORLD_SIZE)[ctx.rank]
        return None, *gO.unbind(-1)

class UnsliceReduceGrad(nn.Module):
    def __init__(self, rank, unslice_state):
        super().__init__()
        self.rank = rank
        self.unslice_state = unslice_state

    def forward(self, w):
        assert isinstance(w, BucketedTensor)
        if len(self.unslice_state) > 0:
            return self.unslice_state[0]
        else:
            new_full_w = UnsliceReduceFn.apply(self.rank, *w._all_views)
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
        out = w.select(-1, self.idx)
        # Make mod.weight.grad work!
        # We only have a the local shard of the gradient in here?
        # out.grad = w.grad_fn.next_functions[self.idx][0].variable.grad
        return out

def make_fsdp_(mod, rank):
    mods = [m for m in mod if isinstance(m, nn.Linear)]
    # Concat all weights together to bucket them together
    full_w = []
    for l in mods:
        w = l.weight
        # Do the sharding by dropping part of the weight locally
        w = w.chunk(WORLD_SIZE)[rank]
        full_w.append(w)

    with torch.no_grad():
        full_w = torch.stack(full_w, dim=-1)
        new_w = BucketedTensor.from_full_tensor(full_w)

    # Update the weights to the new view
    unslice_state = []
    for i, l in enumerate(mods):
        l.weight = nn.Parameter(new_w[i])

        register_parametrization(l, "weight", UnsliceReduceGrad(rank, unslice_state), unsafe=True)
        register_parametrization(l, "weight", Slice(i), unsafe=True)

# ### Quantization part
# class QuantizedTensor(torch.Tensor):
#     @staticmethod
#     def __new__(cls, data, dtype, requires_grad=False):
#         return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=dtype)

#     def __init__(self, data, dtype, requires_grad=False):
#         self._data = data

#     __torch_function__ = torch._C._disabled_torch_function_impl

#     def __repr__(self):
#         # This is (1) sad that we have to special case (2) Parameter printing is so broken
#         if isinstance(self, nn.Parameter):
#             prefix = 'Parameter containing:\n'
#         else:
#             prefix = ''
#         return prefix + f"QuantizedTensor({self._data}, public_dtype={self.dtype})"

#     @classmethod
#     def __torch_dispatch__(cls, func, types, args, kwargs=None):
#         args, kwargs = pytree.tree_map_only(QuantizedTensor, lambda x: x._data, (args, kwargs or {}))
#         # Always downcast everything to bfloat16 for simplicity
#         if pytree.tree_any_only(torch.Tensor, lambda x: x.dtype is torch.bfloat16, args):
#             args, kwargs = pytree.tree_map_only(torch.Tensor, lambda x: x.to(torch.bfloat16) if x.dtype is torch.float32 else x, (args, kwargs))
#         raw_out = func(*args, **kwargs)
#         out = pytree.tree_map_only(torch.Tensor, lambda x: QuantizedTensor(x, torch.float32), raw_out)
#         return out

# def quantize_params_(mod):
#     for l in mod:
#         l.weight = nn.Parameter(QuantizedTensor(l.weight.to(torch.bfloat16), torch.float32))


### TESTING
import torch.distributed as dist
# This is not imported by default, so we need an explicit import
import torch.distributed._functional_collectives
import torch.multiprocessing as mp
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

WORLD_SIZE = 2

def fsdp_main(idx):
    def _print(txt):
        print(f"From {idx}: {txt}")
    setup(idx, WORLD_SIZE)
    torch.manual_seed(idx)
    inp = torch.rand(4, 2000)
    target = torch.randn(4, 2000)

    torch.manual_seed(42)
    mod = nn.Sequential(
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 2000),
    )
    crit = torch.nn.MSELoss()

    _print(f"init w[0]: {mod[0].weight.mean()}")
    make_fsdp_(mod, idx)
    _print(f"fsdp w[0]: {mod[0].weight.mean()}")
    opt = torch.optim.SGD(mod.parameters(), lr=0.01, momentum=0.9)

    for i in range(4):
        _print(f"fw w[0]: {mod[0].weight.mean()}")
        pred = mod(inp)
        l = crit(pred, target)
        _print(f"loss: {l}")

        l.backward()
        if hasattr(mod[0], "parametrizations"):
            _print(f"grad: {mod[0].parametrizations.weight.original.grad.mean()}")
        else:
            _print(f"grad: {mod[0].weight.grad.mean()}")

        opt.step()
        opt.zero_grad()
    _print(f"final weights w[0]: {mod[0].weight.mean()}")

    cleanup()

if __name__ == "__main__":
    mp.spawn(fsdp_main,
        nprocs=WORLD_SIZE,
        join=True)

