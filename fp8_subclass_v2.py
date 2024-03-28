from enum import Enum

import torch
from torch.library import Library
import torch.nn as nn
import torch.nn.functional as F

from torch.utils._pytree import tree_map

class FP8Flavor(Enum):
    E4M3 = 0
    E5M2 = 1

torch.manual_seed(0)

aten = torch.ops.aten

#
# fp8 packing utility for e4m3 format
# TODO(future): align with https://github.com/pytorch/FBGEMM/pull/974/files
# and NVIDIA primitives
# TODO(future): add support for e5m2
#
def float_to_bits8(input_tensor):
    # Ignores all kind of non-normalized values:
    # - Round very small numbers to 0
    # - Raises an error on numbers being too large
    # Extract exponent and sign
    mantissa, exp = torch.frexp(input_tensor)
    sign = (mantissa.sign() + 1) / 2

    # Encore the exponent
    new_exp = exp + (15 // 2)
    # Round small numbers to lowest possible exponent
    # and ensure not wrapping in uint8 so that the check below is sound
    new_exp.clamp_(0, 0xFF)
    enc_exp = new_exp.to(torch.uint8)
    if (enc_exp > 0xF).any():
        raise RuntimeError(f"Exponent value too large when converting {input_tensor}")

    # Assume we can just steal the mantissa data from a regular float?
    new_mantissa = ((input_tensor.view(torch.int32) & 0x700000) >> 20).to(torch.uint8)

    # Generate our fp8 inside a uint8
    output_tensor = (sign.to(torch.uint8) << 7) | (enc_exp << 3) | new_mantissa
    return output_tensor.view(torch.bits8)

def bits8_to_float(input_tensor):
    # Not many ops work with bits8
    input_tensor = input_tensor.view(torch.uint8)
    # Get sign
    input_tensor_sign = (input_tensor & 0x80) >> 7

    # Read exponent to number
    enc_exp = (input_tensor & 0x78) >> 3
    exp = enc_exp.to(torch.int32) - (15 // 2)

    # Provide a dummy mantissa bit with the right sign
    out = torch.ldexp((input_tensor_sign.float() * 2) - 1, exp)

    # Overwrite the mantissa with the original one
    out_int = out.view(torch.int32)
    out_int &= 0xFF800000
    out_int |= (input_tensor & 0x7) << 20

    return out

def mm_fp8(m1, s1, m2, s2, sout):
    # TODO(future): add e4m3/e5m2
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    full_m1 = bits8_to_float(m1) * s1
    full_m2 = bits8_to_float(m2) * s2
    full_out = torch.mm(full_m1, full_m2)
    out = full_out / sout
    return float_to_bits8(out)

def add_fp8(m1, s1, m2, s2, s3):
    # TODO(future): add e4m3/e5m2
    # naive implementation: dq -> op -> q
    # TODO(future): hook up to real kernel
    m1_fp32 = bits8_to_float(m1) * s1
    m2_fp32 = bits8_to_float(m2) * s2
    m3_fp32 = m1_fp32 + m2_fp32
    return float_to_bits8(m3_fp32 / s3)

#
# ATen op placeholders
#

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

# Define our new custom functions
# Since all my Tensors are on CPU, I register everything there.
lib.define("fp32_to_fp8(Tensor t) -> Tensor")
lib.impl("fp32_to_fp8", float_to_bits8, "CPU")

lib.define("fp8_to_fp32(Tensor t) -> Tensor")
lib.impl("fp8_to_fp32", bits8_to_float, "CPU")

lib.define("mm_fp8(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor sout) -> Tensor")
lib.impl("mm_fp8", mm_fp8, "CPU")

lib.define("add_fp8(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor s3) -> Tensor")
lib.impl("add_fp8", add_fp8, "CPU")


class FP8ConstrFunc(torch.autograd.Function):
    """
    A differentiable conversion between fp32 and fp8
    TODO(future): split into two for cleaner code
    """
    @staticmethod
    def forward(ctx, tensor, scale: float=None, flavor=FP8Flavor.E4M3):
        if isinstance(tensor, FP8Tensor):
            ctx.inp_is_fp8 = True
            return torch.ops.aten.fp8_to_fp32(tensor._data) * tensor._scale
        else:
            ctx.inp_is_fp8 = False
            tensor_scaled = tensor / scale
            bits_fp8 = torch.ops.aten.fp32_to_fp8(tensor_scaled)
            return FP8Tensor(bits_fp8, scale, flavor)

    @staticmethod
    def backward(ctx, g):
        # Assume that we always want to scale the gradients
        # back to full precision. We could do something else
        if isinstance(g, FP8Tensor) and not ctx.inp_is_fp8:
            return g.to_fp32(), None, None
        elif ctx.inp_is_fp8:
            return FP8Tensor.from_fp32(g), None, None
        else:
            return g, None, None


class FP8Tensor(torch.Tensor):
    """
    A Python-only FP8 tensor.  Contains:
    * `_data`: the underlying e4m3 data (TODO add e5m2 support)
    * `_scale`: the scale used to scale the original fp32 tensor
    * `_flavor`: either E4M3 or E5M2 (for now, this does not change numerics
        and is only present to demonstrate distinguishing between the flavors
        in the framework)

    The current purpose of this object is 99% to bundle raw data + fp8 metadata
    together for easy passing through PyTorch systems, and 1% to implement
    gradient addition (since that has to happen outside of user code).

    The addition operation is defined inline and uses a naive
    version of stateless scaling. This allows e5m2 gradients to be added.
    TODO(future): verify this is numericaly accurate, optionally replace
    with something better.

    It would probably make sense to also define fp8 path for data shuffling
    ops like cat, transpose, view, etc inline so we don't have to fall back
    to fp32 for them.
    """

    def __new__(cls, data, scale, flavor):
        # This is a non-differentiable constructor!
        assert not data.requires_grad
        assert data.dtype == torch.bits8
        assert scale.dtype == torch.float32
        assert scale.nelement() == 1

        self = torch.Tensor._make_wrapper_subclass(
            cls,
            data.size(),
            strides=data.stride(),
            storage_offset=data.storage_offset(),
            dtype=torch.float32,
            layout=data.layout,
            requires_grad=data.requires_grad,
            device=data.device,
        )
        self._data = data
        self._scale = scale
        self._flavor = flavor

        return self

    def __repr__(self):
        return f"FP8Tensor(flavor={self._flavor}, scale={self._scale}, as_fp32={self.to_fp32()}"

    def to_fp32(self):
        return FP8ConstrFunc.apply(self)

    @classmethod
    def from_fp32(cls, tensor, scale, flavor):
        return FP8ConstrFunc.apply(tensor, scale, flavor)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # Note: unlike many other subclasses, this subclass's only propagates
        # itself for addition (for gradient addition in backward). For all
        # other ops, it self-converts to fp32. The user/framework is
        # assumed to take care of defining where fp8 operations occur in the
        # forward pass and how scaling is calculated. In this example, that is
        # done by the `FP8Linear` class.
        # Vasiliy: the main reason I went with ^ is because NVIDIA is
        # doing stateful delayed scaling, and I don't know of a safe
        # way to enable that without either full program capture or punting it
        # to the user. This prototype takes the "punt it to the user" approach.
        # IMO for now let's just write out the scale stuff manually so we can
        # focus on other things, and revisit later if needed.

        # override addition so we can add e5m2 gradients
        if (
            func is aten.add.Tensor
            and isinstance(args[0], FP8Tensor)
            and isinstance(args[1], FP8Tensor)
        ):
            x1_fp8, x2_fp8 = args[0], args[1]
            print(x1_fp8, x2_fp8)
            # naive scale calculation: max of incoming two scales
            x3_scale = torch.max(x1_fp8._scale, x2_fp8._scale)
            res_bits = torch.ops.aten.add_fp8(
                x1_fp8._data, x1_fp8._scale,
                x2_fp8._data, x2_fp8._scale,
                x3_scale)
            res = FP8Tensor(res_bits, x3_scale, x1_fp8._flavor)
            return res

        # for all other ops, fall back to fp32
        # TODO(future): add support for fp16/bf16

        def maybe_unwrap(t):
            if isinstance(t, FP8Tensor):
                return t.to_fp32()
            return t

        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the FP8Tensor type on the returned tensor
    __torch_function__ = torch._C._disabled_torch_function_impl

class fp8_linear_no_bias(torch.autograd.Function):
    """
    Like F.linear, but with X, W, and Y in fp8
    TODO(future) add logic for bias
    """

    @staticmethod
    def forward(
        ctx,
        x_fp8,
        w_t_fp8,
        fp8_s_out,
        fp8_s_dL_dX,
        fp8_s_dL_dW,
        fp8_s_dL_dY,
    ):
        ctx.save_for_backward(x_fp8, w_t_fp8, fp8_s_dL_dX, fp8_s_dL_dW, fp8_s_dL_dY)

        res_bits = torch.ops.aten.mm_fp8(
            x_fp8._data, x_fp8._scale,
            w_t_fp8._data.t(), w_t_fp8._scale,
            fp8_s_out)

        res = FP8Tensor(res_bits, fp8_s_out, FP8Flavor.E4M3)
        # scale update would also happen here, for now no-op
        return res

    @staticmethod
    def backward(ctx, grad_output):
        x_fp8, w_t_fp8, fp8_s_dL_dX, fp8_s_dL_dW, fp8_s_dL_dY = \
            ctx.saved_tensors

        if not isinstance(grad_output, FP8Tensor):
            grad_output_fp8 = FP8Tensor(
                torch.ops.aten.fp32_to_fp8(grad_output / fp8_s_dL_dY),
                fp8_s_dL_dY,
                FP8Flavor.E5M2)
        else:
            grad_output_fp8 = grad_output

        dL_dX_bits = torch.ops.aten.mm_fp8(
            grad_output_fp8._data, grad_output_fp8._scale,
            w_t_fp8._data, w_t_fp8._scale,
            fp8_s_dL_dX)
        dL_dX_fp8 = FP8Tensor(dL_dX_bits, fp8_s_dL_dX, FP8Flavor.E5M2)

        dL_dW_bits = torch.ops.aten.mm_fp8(
            x_fp8._data.t(), x_fp8._scale,
            grad_output_fp8._data, grad_output_fp8._scale,
            fp8_s_dL_dW).t()
        dL_dW_fp8 = FP8Tensor(dL_dW_bits, fp8_s_dL_dW, FP8Flavor.E5M2)

        # scale update would also happen here, for now no-op
        return dL_dX_fp8, dL_dW_fp8, None, None, None, None



class FP8Linear(torch.nn.Linear):
    """
    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO(future): real scale calculations, for now it's mocked out
        self.register_buffer('fp8_s_in', torch.tensor(1.0))
        self.register_buffer('fp8_s_weight', torch.tensor(1.0))
        self.register_buffer('fp8_s_out', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dX', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dW', torch.tensor(1.0))
        self.register_buffer('fp8_s_dL_dY', torch.tensor(1.0))

    def forward(self, x):
        if not isinstance(x, FP8Tensor):
            x_fp8 = FP8Tensor.from_fp32(x, self.fp8_s_in, FP8Flavor.E4M3)
        else:
            x_fp8 = x
        w_t_fp8 = FP8Tensor.from_fp32(self.weight, self.fp8_s_weight, FP8Flavor.E4M3)

        y_fp8 = fp8_linear_no_bias.apply(
            x_fp8, w_t_fp8, self.fp8_s_out, self.fp8_s_dL_dX,
            self.fp8_s_dL_dW, self.fp8_s_dL_dY)

        # For now, hardcode returning FP8Tensor (propagate as much as we can).
        # This can be changed to return a different dtype, if needed.
        return y_fp8

    @classmethod
    def from_float(cls, mod):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear
        """
        assert mod.bias is None, 'bias support not implemented yet'
        new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        return new_mod

if __name__ == "__main__":

    # test pack/unpack
    print("\nExample of packing/unpacking\n")
    x = torch.randn(2) / 5
    print("original", x)
    out = torch.ops.aten.fp32_to_fp8(x)
    print("fp8 as uint8", out.view(torch.uint8))
    y = torch.ops.aten.fp8_to_fp32(out)
    print("back to fp32", y)

    # test addition
    print("\nExample of addition\n")
    x1_fp32, x1_s = torch.randn(4), torch.tensor(1.0)
    x2_fp32, x2_s = torch.randn(4), torch.tensor(1.0)
    x1_fp8 = FP8Tensor.from_fp32(x1_fp32, x1_s, FP8Flavor.E5M2)
    x2_fp8 = FP8Tensor.from_fp32(x2_fp32, x2_s, FP8Flavor.E5M2)
    x3_fp8 = x1_fp8 + x2_fp8
    print('x1', x1_fp8, '\nx2', x2_fp8, '\nx1+x2', x3_fp8)


    print("\nExample of fp8 linear fw + bw\n")

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 3, bias=False)
            self.fc2 = nn.Linear(3, 4, bias=False)
            self.fc3 = nn.Linear(3, 4, bias=False)
            self.fc4 = nn.Linear(4, 2, bias=False)

        def forward(self, x0):
            x1 = self.fc1(x0)
            x2 = self.fc2(x1)
            x3 = self.fc3(x1)
            # test gradient addition
            # Note: cat happens in fp32, for now
            c = torch.cat([x2, x3])
            x4 = self.fc4(c)
            return x4

    m = M()
    m.fc1 = FP8Linear.from_float(m.fc1)
    m.fc2 = FP8Linear.from_float(m.fc2)
    m.fc3 = FP8Linear.from_float(m.fc3)
    m.fc4 = FP8Linear.from_float(m.fc4)

    print(m)

    x = FP8Tensor.from_fp32(torch.randn(1, 2), torch.tensor(1.0), FP8Flavor.E4M3)
    y = m(x)
    print(y)
    s = y.sum()
    print('before grad', m.fc1.weight.grad, m.fc2.weight.grad, m.fc3.weight.grad, m.fc4.weight.grad)
    s.backward()
    print('after grad', m.fc1.weight.grad, m.fc2.weight.grad, m.fc3.weight.grad, m.fc4.weight.grad)
