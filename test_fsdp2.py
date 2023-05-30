import torch
from torch.utils import _pytree as pytree
from torch.autograd import Function, grad
from torch import nn
from torch.nn.utils.parametrize import register_parametrization
import contextlib
import os


########################################################################
### Utils
########################################################################

def print_curr_mem(txt):
    print(f"Current mem at {txt}: {torch.cuda.memory_allocated() // 1024 // 1024}")
    pass

@contextlib.contextmanager
def print_peak_mem(name):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    yield
    extra_mem = (torch.cuda.max_memory_allocated() - base_mem) // 1024 // 1024
    print(f"Peak mem {name}: {torch.cuda.max_memory_allocated() // 1024 // 1024} -> delta: {extra_mem}MB")


########################################################################
### Bucketting
########################################################################

class BuckettingReparam(nn.Module):
    def __init__(self, offset, nelem, size, full):
        super().__init__()
        self.offset = offset
        self.size = size
        self.nelem = nelem
        self.full = full

    def right_inverse(self, slice):
        if self.offset == 0:
            return self.full
        else:
            return torch.tensor(())

    def forward(self, _):
        return self.full.narrow(0, self.offset, self.nelem).view(self.size)

def bucket_together_(mod, fqns):
    all_slices = []
    all_mods = []
    all_names = []
    for fqn in fqns:
        *mods, p_name = fqn.split(".")
        m = mod
        for mod_name in mods:
            m = getattr(m, mod_name)

        p = getattr(m, p_name)
        assert p.storage_offset() == 0 and p.is_contiguous(), "Could be lifted but makes flat indexing easier"
        all_slices.append(p)
        all_mods.append(m)
        all_names.append(p_name)

    full = torch.cat(tuple(s.view(-1) for s in all_slices))

    offset = 0
    for m, name, s in zip(all_mods, all_names, all_slices):
        nelem = s.nelement()
        reparam = BuckettingReparam(offset, nelem, s.size(), full)
        register_parametrization(m, name, reparam, unsafe=True)

        offset += nelem

## Test
torch.manual_seed(42)
inp = torch.rand(10, 200)
mod = nn.Sequential(
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
)
crit = torch.nn.MSELoss()

ref_out = mod(inp)
print("Bucketting:")
print(mod)
bucket_together_(mod, ['0.weight', '2.weight'])
print(mod)
new_out = mod(inp)
assert (ref_out - new_out).abs().max() < 1e-5, "Bad bucketting"


########################################################################
### Sharding
########################################################################

class ShardingReparam(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.idxs = list(range(WORLD_SIZE))

    def right_inverse(self, full):
        # local = torch.distributed._functional_collectives.reduce_scatter(full, "sum", ctx.idxs)
        # Make it work locally
        local = full.chunk(WORLD_SIZE)[self.rank]
        # Clone to make sure we don't keep the full storage alive
        return local.clone()

    def forward(self, local):
        # full = torch.distributed._functional_collectives.all_gather_tensor(local, 0, self.idxs,"huh")
        # Make it work locally
        repeats = [WORLD_SIZE] + [1] * (local.ndim - 1)
        full = local.repeat(repeats)
        return full

def shard_param_(mod, fqn):
    *mods, p_name = fqn.split(".")
    m = mod
    for mod_name in mods:
        m = getattr(m, mod_name)

    register_parametrization(m, p_name, ShardingReparam(rank), unsafe=True)


## Test
torch.manual_seed(42)
inp = torch.rand(10, 200)
mod = nn.Sequential(
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
)
crit = torch.nn.MSELoss()

# Make it work locally
rank = 0
WORLD_SIZE = 4

# Make our fake repeat-based comms "correct"
with torch.no_grad():
    vals = mod[0].weight.chunk(WORLD_SIZE)
    for i in range(1, len(vals)):
        vals[i].copy_(vals[0])

ref_out = mod(inp)
print("Sharding:")
print(mod)
print(ref_out.mean())
shard_param_(mod, '0.weight')
print(mod)
new_out = mod(inp)
print(new_out.mean())
assert (ref_out - new_out).abs().max() < 1e-5, "Bad sharding"


########################################################################
### Bucketting + Sharding
########################################################################

torch.manual_seed(42)
inp = torch.rand(10, 200)
mod = nn.Sequential(
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
)
crit = torch.nn.MSELoss()

# Make it work locally
rank = 0
WORLD_SIZE = 4

# Make our fake repeat-based comms "correct"
with torch.no_grad():
    vals = mod[0].weight.chunk(WORLD_SIZE)
    for i in range(1, len(vals)):
        vals[i].copy_(vals[0])

ref_out = mod(inp)
print("Bucketting + Sharding:")
print(mod)
print(ref_out.mean())
bucket_together_(mod, ['0.weight', '2.weight'])
shard_param_(mod, '0.parametrizations.weight.original')
print(mod)
new_out = mod(inp)
print(new_out.mean())
assert (ref_out - new_out).abs().max() < 1e-5, "Bad sharding"


########################################################################
### Skip saving param
########################################################################

class DropHooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, base_getter):
        def pack(tensor):
            t = tensor._base if tensor._base is not None else tensor
            if getattr(t, "_drop_me", False):
                ref = getattr(t, "_ref", [])
                t._ref = ref
                return ref, tensor.size(), tensor.stride(), tensor.storage_offset()
            return tensor

        def unpack(data):
            if isinstance(data, torch.Tensor):
                return data
            else:
                ref, size, stride, storage_offset = data
                if len(ref) == 0:
                    ref.append(base_getter())
                # We could record view ops if we don't want as_strided
                new_t = ref[0].as_strided(size, stride, storage_offset)
                return new_t

        super().__init__(pack, unpack)

class DroppingReparam(nn.Module):
    def right_inverse(self, inp):
        return inp

    def forward(self, inp):
        t = inp._base if inp._base is not None else inp
        t._drop_me = True
        return inp

def drop_param_(mod, fqn):
    *mods, p_name = fqn.split(".")
    m = mod
    for mod_name in mods:
        m = getattr(m, mod_name)

    register_parametrization(m, p_name, DroppingReparam(), unsafe=True)

    # Assume that this is the last reparam and so this getattr is the
    # right way to recompute!
    def getter():
        return getattr(m, p_name)
    return DropHooks(getter)

## Test
torch.manual_seed(42)
inp = torch.rand(10, 200, requires_grad=True)
mod = nn.Sequential(
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
    nn.ReLU(),
    nn.Linear(200, 200),
)
crit = torch.nn.MSELoss()

ref_out = mod(inp)
ref_g, = grad(ref_out.sum(), inp)
print("Buffer dropping:")
ctx = drop_param_(mod, '2.weight')
with ctx:
    new_out = mod(inp)
    new_g, = grad(new_out.sum(), inp)
assert (ref_out - new_out).abs().max() < 1e-5, "Buffer dropping"
assert (ref_g - new_g).abs().max() < 1e-5, "Buffer dropping grad"


########################################################################
### Sharding + Dropping
########################################################################

torch.manual_seed(42)
inp = torch.rand(2, 2000, device="cuda", requires_grad=True)
print_curr_mem("Bef module")
mod = nn.Sequential(
    nn.Linear(2000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 2000),
    nn.ReLU(),
    nn.Linear(2000, 2000),
    nn.ReLU(),
).cuda()
print_curr_mem("After module")
crit = torch.nn.MSELoss()

# Make it work locally
rank = 0
WORLD_SIZE = 100

# Make our fake repeat-based comms "correct"
with torch.no_grad():
    for i in [0, 2, 4]:
        vals = mod[i].weight.chunk(WORLD_SIZE)
        for i in range(1, len(vals)):
            vals[i].copy_(vals[0])

# mem warmup
mod(inp).sum().backward()

print("Buffer Sharding + Dropping:")
print(mod)
print_curr_mem("Bef shard")
shard_param_(mod, '0.weight')
print_curr_mem("0 sharded")
shard_param_(mod, '2.weight')
print_curr_mem("2 sharded")


@print_peak_mem("ref")
def do():
    ref_out = mod(inp)
    ref_g, = grad(ref_out.sum(), inp)
    return ref_out, ref_g
ref_out, ref_g = do()

ctx = drop_param_(mod, '0.weight')
print(mod)

@print_peak_mem("new")
def do():
    with ctx:
        new_out = mod(inp)
        new_g, = grad(new_out.sum(), inp)
    return new_out, new_g
new_out, new_g = do()

assert (ref_out - new_out).abs().max() < 1e-5, "Buffer sharding + dropping"
assert (ref_g - new_g).abs().max() < 1e-5, "Buffer sharding + dropping grad"

# Buffer Sharding + Dropping:
# Sequential(
#   (0): Linear(in_features=2000, out_features=2000, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=2000, out_features=2000, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=2000, out_features=2000, bias=True)
#   (5): ReLU()
# )
# Current mem at Bef shard: 112
# Current mem at 0 sharded: 96
# Current mem at 2 sharded: 80
# Peak mem ref: 113 -> delta: 33MB
# Sequential(
#   (0): ParametrizedLinear(
#     in_features=2000, out_features=2000, bias=True
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0): ShardingReparam()
#         (1): DroppingReparam()
#       )
#     )
#   )
#   (1): ReLU()
#   (2): ParametrizedLinear(
#     in_features=2000, out_features=2000, bias=True
#     (parametrizations): ModuleDict(
#       (weight): ParametrizationList(
#         (0): ShardingReparam()
#       )
#     )
#   )
#   (3): ReLU()
#   (4): Linear(in_features=2000, out_features=2000, bias=True)
#   (5): ReLU()
# )
# Peak mem new: 97 -> delta: 17MB



