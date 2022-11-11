import torch
from torch.utils import _pytree as pytree
from torch.autograd import Function

# This example is showing how to implement a QuantizedTensor and how to get it to interact with
# autograd smootly.
# This is ONLY the QuantizedTensor which does a few things:
#  - Hold only the low precision data
#  - Route implementation to the right custom kernel when available
#  - Perform type promotion to use fallbackward when custom kernel not available
#  - "pretends" to be a full precision floating point Tensor to the outside world

class Quantizer(Function):
    @staticmethod
    def forward(ctx, base):
        # Just to do the quantization
        out_data = base.to(torch.int8)
        return QuantizedTensor(out_data, base.dtype)

    @staticmethod
    def backward(ctx, gO):
        # Assume we always do gradient computation in full precision
        return gO

# Small util, should exist somewhere else?
def compare_dtype(d1, d2):
    if d1.is_floating_point:
        return d1
    elif d2.is_floating_point:
        return d2
    else:
        assert False, "NYI"

class QuantizedTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data, dtype, requires_grad=False):
        # This constructor can ONLY create leaf Tensors wrt autograd.
        # Use QuantizedTensor.from_tensor(t) to get a non-leaf Tensor wrt autograd.
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), dtype=dtype, requires_grad=requires_grad)

    def __init__(self, data, dtype, requires_grad=False):
        self._data = data

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __repr__(self):  # Zero out missing values for printing
        autograd_info = f", grad_fn={self.grad_fn}" if self.grad_fn else f", requires_grad=True" if self.requires_grad else ""
        return f"QuantizedTensor({self._data}, public_dtype={self.dtype}{autograd_info})"

    @classmethod
    def from_tensor(cls, base):
        # This is a differentiable function!!
        return Quantizer.apply(base)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Basic implementation that will need refinement based on what should be upcasted or not
        # similar to amp.
        # For now, just do the compute in the highest precision of any input and requantize
        # like the first one. While ignoring all non-floating point dtypes.
        base_qt_tensor = None
        for a in args:
            if isinstance(a, QuantizedTensor):
                base_qt_tensor = a
                break
        assert base_qt_tensor is not None
        inp_dtype = base_qt_tensor._data.dtype
        out_public_dtype = base_qt_tensor.dtype
        # Unpack QuantizedTensor
        args, kwargs = pytree.tree_map_only(QuantizedTensor, lambda x: x._data, (args, kwargs or {}))
        # Get highest dtype
        highest_type = inp_dtype
        def check_type(t):
            nonlocal highest_type
            if t.dtype.is_floating_point and compare_dtype(t.dtype, highest_type):
                highest_type = t.dtype
        pytree.tree_map_only(torch.Tensor, check_type, (args, kwargs))
        # Promote everything to the right dtype
        args, kwargs = pytree.tree_map_only(torch.Tensor, lambda x: x.to(highest_type) if x.dtype.is_floating_point else x, (args, kwargs))
        # Run the original function with the new dtype
        # This can also be a custom kernel if you need
        raw_out = func(*args, **kwargs)
        # Rewrap everything back
        # Since we're below autograd, we don't need to use from_tensor
        def repack(t):
            if t.dtype is highest_type:
                if highest_type.is_floating_point:
                    # Requantize back to input dtype if we computed in float
                    return QuantizedTensor(t.to(inp_dtype), out_public_dtype)
                else:
                    # Otherwise keep it as-is
                    return QuantizedTensor(t, out_public_dtype)
            # Just a hack for sum that has higher precision result, shouldn't happen if you have
            # custom kernels
            elif func is torch.ops.aten.sum.default and t.dtype is torch.int64:
                return QuantizedTensor(t, out_public_dtype)
            else:
                return t
        out = pytree.tree_map_only(torch.Tensor, repack, raw_out)
        return out


inp = torch.randint(0, 100, (2,), dtype=torch.float, requires_grad=True)
qt = QuantizedTensor.from_tensor(inp)
print("Input 1")
print(qt)

(qt * 3).sum().backward(retain_graph=True)
print("Raw input 1's grad")
print(inp.grad)

qt2 = QuantizedTensor.from_tensor(torch.randint(0, 100, (2,), dtype=torch.float)).requires_grad_()
print("Input 2")
print(qt2)

(qt2 * qt).sum().backward()
print("Input 2's grad")
print(qt2.grad)
print("Raw input 1's grad")
print(inp.grad)
