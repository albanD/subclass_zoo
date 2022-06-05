import gc
import torch
from torch import Tensor
from torch.utils._pytree import tree_map
import unittest

# Currently in discussion with PyTorch:
# https://discuss.pytorch.org/t/training-with-custom-quantized-datatype/152132


class QuantizedTensor(Tensor):
    @staticmethod
    def __new__(cls, t):
        """ 
        Inspired by NegativeTensor by albanD
        https://github.com/albanD/subclass_zoo/blob/main/negative_tensor.py
        A 1-D tensor in custom 12-bit datatype. 
        Performs an in-place quantization and then re-sizes to the exact size.
        Drawback is that tensor needs to be resized - would be ideal to avoid extra allocation and memory fragmentation."""

        # not compositional
        assert type(t) is Tensor

        assert not t.requires_grad or not torch.is_grad_enabled()

        QUANT_DTYPE_SIZE = 1.5  # each quantized element is 12 bits

        # quantize the tensor
        # TODO: for now set to a memory pattern
        new_size = t.numel() * QUANT_DTYPE_SIZE
        new_size_elems = int(new_size / t.element_size())
        dummy_data = 5.5 * torch.ones(new_size_elems, dtype=t.dtype)  # 5.5 is arbitrary

        # https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html
        # t.resize_(new_size_elems) does not use - If the number of elements is smaller, the underlying storage is not changed.
        # t.copy_(dummy_data)
        t.set_(dummy_data.storage(), storage_offset=0, size=t.size(), stride=t.stride())
        return Tensor._make_subclass(cls, t)

    def __init__(self, elem):
        self.elem = elem

    def __repr__(self):
        raise RuntimeError("not implemented")

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(x):
            return x.elem if isinstance(x, QuantizedTensor) else x
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs)

        def wrap(x):
            return QuantizedTensorFunction.apply(x) if isinstance(x, torch.Tensor) else x

        if func._overloadpacket == torch.ops.aten._to_copy:
            # TODO
            print("aten._to_copy: Trying to copy QuantizedTensor")
            return tree_map(wrap, func(*args, **kwargs))
        elif func._overloadpacket == torch.ops.aten.detach:
            print("aten.detach: Trying to detach QuantizedTensor")
            return tree_map(wrap, func(*args, **kwargs))
        else:
            raise RuntimeError("TODO: operator not implemented yet")

# Use this to construct QuantizedTensor.


class QuantizedTensorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Exact type matches as QuantizedTensor is not compositional
        if type(input) is QuantizedTensor:
            # If we are passed a QuantizedTensor, we can simply alias it as
            # a normal tensor and return it.
            return torch.Tensor._make_subclass(torch.Tensor, input)
        elif type(input) is Tensor:
            return QuantizedTensor(input)
        else:
            raise AssertionError("QuantizedTensor is not yet compositional")

    @staticmethod
    def backward(ctx, grad):
        return QuantizedTensorFunction.apply(grad)


class QuantizedModuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, w_quantized):
        # will eventually be some actual computation
        return torch.zeros(128, 128, dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_output):
        # will eventually be some actual computation
        grad_input = torch.zeros(128, 128, dtype=grad_output.dtype, device=grad_output.device)
        grad_w = torch.zeros(128, 384, dtype=torch.bfloat16, device=grad_output.device)
        return grad_input, grad_w


class QuantizedModule(torch.nn.Module):
    def __init__(self):
        super(QuantizedModule, self).__init__()
        # will eventually be torch.<some custom dtype>
        # See question 1
        quantized_tensor = QuantizedTensor(torch.zeros(128, 384, dtype=torch.float32))
        self.w_quantized = torch.nn.Parameter(quantized_tensor)
        self.w_fp32 = torch.zeros(128, 384, requires_grad=False)

    def forward(self, input):
        return QuantizedModuleFunction.apply(input, self.w_quantized)


def custom_optimizer(w_quantized, w_fp32, grad):
    """ will eventually update w_fp32 and w_quantized and zero out grad"""
    pass


class QuantizedTensorTest(unittest.TestCase):
    def test_training_loop(self):
        input = torch.zeros((128, 128), requires_grad=True, dtype=torch.bfloat16)
        model = QuantizedModule()
        for i in range(10):
            ans = model(input)
            loss = ans.sum()
            loss.backward()
            if model.w_quantized.grad.dtype != torch.bfloat16:
                # see question 2
                print(f"gradient dtype expected to be torch.bfloat16 but is instead {model.w_quantized.grad.dtype}")
            if (i + 1) % 2 == 0:
                # update weights every 2 microbatches
                custom_optimizer(model.w_quantized, model.w_fp32, model.w_quantized.grad)

    def test_to_bfloat16(self):
        qt = QuantizedTensorFunction.apply(torch.zeros(128, 128))
        qt.to(torch.bfloat16)

    def test_to_with_parameter(self):
        """ Fails with RuntimeError: set_storage is not allowed on a Tensor created from .data or .detach()."""
        qt = torch.nn.Parameter(QuantizedTensorFunction.apply(torch.zeros(128, 128)))
        qt.to(torch.bfloat16)
        print(qt)
