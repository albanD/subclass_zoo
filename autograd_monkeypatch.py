from torch.overrides import TorchFunctionMode
import torch.nn.functional

class BuggyDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p=0.5):
        print("forward")
        return x

    @staticmethod
    def backward(ctx, grad_output):
        print("backward")
        return grad_output, None


class AutogradMonkeypatch(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}
        if func is torch.nn.functional.dropout:
            return BuggyDropout.apply(*args)
        return func(*args, **kwargs)

with AutogradMonkeypatch():
    torch.nn.functional.dropout(torch.randn(4, requires_grad=True)).sum().backward()
