import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

class NanDetect(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        res = func(*args, **kwargs)
        flat_res, _ = tree_flatten(res)

        for t in flat_res:
            if not torch.is_tensor(t):
                continue
            try:
                if (t != t).any():
                    raise RuntimeError(
                        f"Function {func}(*{args}, **{kwargs}) " "returned a NaN"
                    )
            except NotImplementedError:
                pass
        return res

a = torch.tensor([0.,])
print(a.div(a))

# This will raise
# RuntimeError: Function aten.div.Tensor(*(tensor([0.]), tensor([0.])), **{}) returned a NaN
with NanDetect():
    print(a.div(a))

