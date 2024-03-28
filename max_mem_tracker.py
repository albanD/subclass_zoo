import torch
from torch.utils._pytree import tree_map_only
from torch.utils._python_dispatch import TorchDispatchMode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.weak import WeakIdKeyDictionary
import weakref
import math

# Track all the memory being used by Tensors.
# Only max is tracked but others can be added.
MEMORY_USE = WeakIdKeyDictionary()
MEMORY_MAX = 0
# Minimum allocation size 
PYTORCH_MIN_ALLOCATE = 2**9

def update_stats():
    global MEMORY_MAX
    curr_use = 0
    for k, v in MEMORY_USE.items():
        curr_use += math.ceil(k.size() * k.element_size()/PYTORCH_MIN_ALLOCATE) * PYTORCH_MIN_ALLOCATE

    if MEMORY_MAX < curr_use:
        MEMORY_MAX = curr_use

# Should be called on every Tensor created
def track(t:torch.Tensor):
    def cb(_):
        update_stats()
    st = t.untyped_storage()
    wt = weakref.ref(st, cb)
    MEMORY_USE[st] = wt
    update_stats()

# Use this Mode to call track on every Tensor being created by functions
class MemoryTrackingMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        res = func(*args, **kwargs or {})

        tree_map_only(torch.Tensor, track, res)
        return res


if __name__ == "__main__":
    # Use FakeTensorMode to run the code without any actual data
    with FakeTensorMode(), MemoryTrackingMode():
        def f(a):
            b = a * 10
            d = b + 3
            return d

        a = torch.rand(100)
        f(a)
        f(a)
        print(f"Just f: {MEMORY_MAX}")
        c = f(a)
        c = f(a)
        print(f"f with return: {MEMORY_MAX}")


