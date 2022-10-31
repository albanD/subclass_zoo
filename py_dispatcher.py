from torch._dispatch.python import enable_python_dispatcher, no_python_dispatcher
import torch

@torch.ops.aten.sub.Tensor.py_impl(torch._C.DispatchKey.CPU)
def my_sub(x, y):
    print("Hello")
    # This private API permits dispatcher to return to Python dispatcher if
    # there are internal dispatches.
    # return torch.ops.aten.sub.Tensor._op_dk(torch._C.DispatchKey.CPU, x, y)
    with no_python_dispatcher():
        return torch.ops.aten.sub.Tensor(x, y)

x = torch.tensor(2)
with enable_python_dispatcher():
    print(torch.sub(x, x))

# Hack to apply it globally
ctx = enable_python_dispatcher()
ctx.__enter__()

print(torch.sub(x, x))
