import gc
import os
import torch
from torch import Tensor
from torch.utils._pytree import tree_map
import unittest

# Without this, I see a "Descriptors cannot not be created directly." error
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


if True:  # so that formatting tool does not reorder
    from torch.testing._internal.common_utils import run_tests, TestCase

# Currently in discussion with PyTorch:
# https://discuss.pytorch.org/t/training-with-custom-quantized-datatype/152132


class QuantizedTensor(Tensor):
    @staticmethod
    def __new__(cls, size, dtype, raw_data):
        # See __init__ function for details on how data are saved
        # Autograd should always be already handled by the time we get here!
        assert not raw_data.requires_grad or not torch.is_grad_enabled()
        return Tensor._make_wrapper_subclass(cls, size,
                                             dtype=dtype,
                                             layout=raw_data.layout,
                                             device=raw_data.device)

    def __repr__(self):
        info = []
        info.append(f"raw_data={repr(self.raw_data)}")
        info.append(f"size={repr(self.size())}")
        if self.dtype is not torch.float32:
            info.append(f"dtype={repr(self.dtype)}")
        if self.requires_grad:
            if self.grad_fn:
                info.append(f"grad_fn={repr(self.grad_fn)}")
            else:
                info.append(f"requires_grad={repr(self.requires_grad)}")
        return "QuantizedTensor(" + ", ".join(info) + ")"

    def __init__(self, size, dtype, raw_data):
        # Note that this contructor is ONLY for packing an already quantized
        # data, use `QuantizedTensor.from_tensor(inp)` to create a quantized Tensor
        # based on a regular one.
        # There are two Tensors involved in this class:
        # - self: a Tensor with no data (create via wrapper subclass) that
        #         we use to "pretend" to be full precision dtype.
        # - self.raw_data: a Tensor that hold the quantized data.
        self.raw_data = raw_data

    @classmethod
    def from_tensor(cls, inp):
        # Creates a new quantized Tensor based on a regular Tensor
        QUANT_DTYPE_SIZE = 1.5  # each quantized element is 12 bits
        # quantize the tensor
        new_size = inp.numel() * QUANT_DTYPE_SIZE
        new_size_elems = int(new_size / inp.element_size())
        # new_ones to copy all other properties from inp
        raw_data = 5.5 * inp.new_ones(new_size_elems)  # 5.5 is arbitrary
        return cls(inp.size(), inp.dtype, raw_data)

    def to_tensor(self, dtype=None):
        # Get a plain Tensor with the same content as this one in the given dtype
        # Arbitrary impl here, but depends on how it is quantized
        out = torch.zeros(self.size(), device=self.raw_data.device, dtype=dtype)
        # Should dequantize self.raw_data and put that in out
        # Just put some data in it
        out.copy_(self.raw_data.view(-1)[0])
        return out

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def call_on_raw(cls, func, args, kwargs):
        # This function should only be used for function that don't change
        # the size/dtype of the input. Special cased for `t` that transpose
        # the size
        inp_size = None
        inp_dtype = None

        def unwrap(x):
            nonlocal inp_size, inp_dtype
            if isinstance(x, cls):
                inp_size = x.size()
                if func is torch.ops.aten.t.default:
                    assert len(inp_size) == 2
                    inp_size = (inp_size[1], inp_size[0])
                inp_dtype = x.dtype
                return x.raw_data
            else:
                return x

        def wrap(x):
            assert inp_size is not None and inp_dtype is not None
            if isinstance(x, torch.Tensor):
                return cls(inp_size, inp_dtype, x)
            else:
                return x

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func._overloadpacket == torch.ops.aten._to_copy:
            # TODO
            print("aten._to_copy: Trying to copy QuantizedTensor")
            if "dtype" in (kwargs or {}):
                raise RuntimeError("Cross dtype to is not supported yet")
            else:
                return cls.call_on_raw(func, args, kwargs)
        elif func._overloadpacket == torch.ops.aten.detach:
            print("aten.detach: Trying to detach QuantizedTensor")
            return cls.call_on_raw(func, args, kwargs)
        elif func is torch.ops.aten.mm.default:
            return mm_impl(*args, **kwargs)
        elif func is torch.ops.aten.t.default:
            return cls.call_on_raw(func, args, kwargs)
        else:
            raise RuntimeError(f"TODO: operator {func} not implemented yet")


def mm_impl(self, other):
    # Only one side works for now, others can be added later
    assert type(self) is Tensor and type(other) is QuantizedTensor
    # Not sure how to do the op with quantized Tensor?
    # Arbitrarily choose to follow self's precision
    return self.mm(other.to_tensor(self.dtype))


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


class QuantizedModule(torch.nn.Module):
    def __init__(self):
        super(QuantizedModule, self).__init__()
        # will eventually be torch.<some custom dtype>
        # See question 1
        quantized_tensor = QuantizedTensor.from_tensor(torch.zeros(128, 384, dtype=torch.float32))
        self.w_quantized = torch.nn.Parameter(quantized_tensor)
        self.w_fp32 = torch.zeros(128, 384, requires_grad=False)

    def forward(self, input):
        return input.mm(self.w_quantized)


def custom_optimizer(w_quantized, w_fp32, grad):
    """ will eventually update w_fp32 and w_quantized and zero out grad"""
    pass


class QuantizedTensorTest(TestCase):
    def test_training_loop(self):
        input = torch.zeros((128, 128), requires_grad=True, dtype=torch.bfloat16)
        model = QuantizedModule()
        for i in range(10):
            ans = model(input)
            loss = ans.sum()
            loss.backward()
            if (i + 1) % 2 == 0:
                # update weights every 2 microbatches
                custom_optimizer(model.w_quantized, model.w_fp32, model.w_quantized.grad)

    def test_to_bfloat16(self):
        qt = QuantizedTensor.from_tensor(torch.zeros(128, 128, dtype=torch.bfloat16))
        print(qt)

    def test_to_with_parameter(self):
        """ Fails with RuntimeError: set_storage is not allowed on a Tensor created from .data or .detach()."""
        qt = torch.nn.Parameter(QuantizedTensor.from_tensor(torch.zeros(128, 128, dtype=torch.bfloat16)))
        print(qt)

    def test_grad(self):
        # compute gradient for an op that is a mix of quantized and not quantized
        t1 = QuantizedTensor.from_tensor(torch.rand(2, 3)).requires_grad_(True)
        t2 = torch.rand(4, 2, requires_grad=True)
        out = t2.mm(t1)
        out.sum().backward()
        self.assertEqual(out.size(), (4, 3))

        # Do the same thing with plain Tensors
        new_t1 = t1.detach().to_tensor().requires_grad_()
        new_t2 = t2.detach().clone().requires_grad_()
        new_out = new_t2.mm(new_t1)
        new_out.sum().backward()
        self.assertEqual(new_out.size(), (4, 3))

        # Make sure the gradients are the same
        self.assertEqual(new_t1.grad, t1.grad)
        self.assertEqual(new_t2.grad, t2.grad)


if __name__ == "__main__":
    run_tests()
