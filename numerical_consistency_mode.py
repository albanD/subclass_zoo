import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from utils import no_dispatch


# The goal of this mode is to check that two device are consistent
# in what they compute.
# We do NOT run the two models in parallel, we only branch at the op
# level to make sure the two branch don't slowly diverge.

def as_tuple(o):
    if isinstance(o, tuple):
        return o
    else:
        return (o,)

class ConsistentWithCPUMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs):
        orig_out = func(*args, **kwargs)

        # Run the same thing on CPU
        # and convert original outputs to CPU
        cpu_args, cpu_kwargs, orig_cpu_out = tree_map_only(torch.Tensor,
                                                           lambda x: x.cpu(),
                                                           (args, kwargs, orig_out))
        cpu_out = func(*cpu_args, **cpu_kwargs)

        # Make sure the output is close enough!
        for orig, cpu in zip(as_tuple(orig_cpu_out), as_tuple(cpu_out)):
            with no_dispatch():
                torch.testing.assert_close(orig, cpu)


        return orig_out

t = torch.rand(100, device="cuda")

# This should work just fine!
with ConsistentWithCPUMode():
    t2 = t + 2
    t3 = t2.norm()
    t4 = t2 / t3


# Let's break some cuda impl!
def my_new_norm_is_actually_a_mean(t):
    return t.mean()

aten = torch.library.Library("aten", "IMPL")
aten.impl("linalg_vector_norm", my_new_norm_is_actually_a_mean, "CUDA")

# We should see that the impl is not correct anymore!
with ConsistentWithCPUMode():
    t2 = t + 2
    try:
        t3 = t2.norm()
    except AssertionError as e:
        print("Norm evaluation failed as expected:")
        print(e)
    else:
        raise AssertionError("Error was not raised!")

