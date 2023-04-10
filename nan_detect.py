import torch
from torch.utils._python_dispatch import TorchDispatchMode

class NanDetect(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)

        if any(res != res):
            raise RuntimeError(f"Function {func}(*{args}, **{kwargs}) "
                               "returned a NaN")
        return res

a = torch.tensor([0.,])
print(a.div(a))

# This will raise
# RuntimeError: Function aten.div.Tensor(*(tensor([0.]), tensor([0.])), **{}) returned a NaN
with NanDetect():
    print(a.div(a))

