from torch._dispatch.python import enable_python_dispatcher, no_python_dispatcher
import torch

# TODO: See https://github.com/pytorch/pytorch/issues/88109 for why
# you have to use BackendSelect here and CUDA doesn't work
@torch.ops.aten.randn.default.py_impl(torch._C.DispatchKey.BackendSelect)
def randn(size, device=None, **kwargs):
    with no_python_dispatcher():
        r = torch.ops.aten.randn.default(size, device='cpu', **kwargs)
    return r.to(device)

# TODO: do the rest of the random functions

# Hack to apply it globally
ctx = enable_python_dispatcher()
ctx.__enter__()

torch.manual_seed(0)
x = torch.randn(10, device='cpu')
torch.manual_seed(0)
y = torch.ops.aten.randn.default([10], device='cuda')
torch.testing.assert_close(x, y.cpu())
