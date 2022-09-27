import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
import itertools

# cribbed from https://github.com/albanD/subclass_zoo/blob/main/logging_mode.py

class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s

def fmt(t: object) -> str:
    if isinstance(t, torch.Tensor):
        return Lit(f"torch.tensor(..., size={tuple(t.shape)}, dtype={t.dtype}, device='{t.device}')")
    else:
        return t

class EnhancedErrorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args, kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            fmt_args = ", ".join(
                itertools.chain(
                    (repr(tree_map(fmt, a)) for a in args),
                    (f"{k}={tree_map(fmt, v)}" for k, v in kwargs.items()),
                )
            )
            msg = f"...when running {func}({fmt_args})"
            # https://stackoverflow.com/questions/17677680/how-can-i-add-context-to-an-exception-in-python
            msg = f'{ex.args[0]}\n{msg}' if ex.args else msg
            ex.args = (msg,) + ex.args[1:]
            raise

with EnhancedErrorMode():
    torch.matmul(torch.randn(3), torch.randn(4, 5))
