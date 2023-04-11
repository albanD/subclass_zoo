import torch
from torch.library import Library
from torch.utils._python_dispatch import TorchDispatchMode

from torch.utils._pytree import tree_map_only

aten = torch.ops.aten

# This example is aimed at showing the two levels of fp8 Tensor and how they
# interact with autograd. In particular, the user facing API is kept simple
# and explicit to make this example easier to follow.
# Other high level API can be used if we want to be closer to AMP or at Module level.

# We use a global scope tracker for scales with user manual reset.
FP8_STATS_TRACKING_MODE = None

class FP8StatsTracker():
    def __init__(self):
        self.scales = []
        self.curr_idx = -1

    def reset(self):
        self.curr_idx = 0

    def get_next_scale(self, func):
        if len(self.scales) == self.curr_idx:
            self.scales.append((func, torch.tensor(1.)))

        self.curr_idx += 1
        f, s = self.scales[self.curr_idx - 1]
        assert f is func
        return s

    def update_scale(self, new_bits8):
        # I don't know what the update rule is so doing exponential
        # averaging with 0.9 on max value in input tensor
        func, curr_val = self.scales[self.curr_idx - 1]
        new_out = bits8_to_float(new_bits8) * curr_val
        new_val = new_out.max()
        updated_val = 0.9 * curr_val + 0.1 * new_val
        self.scales[self.curr_idx - 1] = (func, updated_val)

    @classmethod
    def get(cls):
        assert FP8_STATS_TRACKING_MODE is not None
        return FP8_STATS_TRACKING_MODE

    def __enter__(self):
        global FP8_STATS_TRACKING_MODE
        assert FP8_STATS_TRACKING_MODE is None
        FP8_STATS_TRACKING_MODE = self
        return self

    def __exit__(self, *_):
        global FP8_STATS_TRACKING_MODE
        assert FP8_STATS_TRACKING_MODE is self
        FP8_STATS_TRACKING_MODE = None

# Simple custom Function to be able to have a differentiable constructor
class FP8ConstrFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        if isinstance(tensor, FP8Tensor):
            ctx.inp_is_fp8 = True
            return bits8_to_float(tensor._data) * tensor._scale
        else:
            ctx.inp_is_fp8 = False
            d, s = torch.ops.aten.fp8_from_fp32(tensor)
            return FP8Tensor(d, s)

    @staticmethod
    def backward(ctx, g):
        # Assume that we always want to scale the gradients
        # back to full precision. We could do something else
        if isinstance(g, FP8Tensor) and not ctx.inp_is_fp8:
            return g.to_fp32()
        elif ctx.inp_is_fp8:
            return FP8Tensor.from_fp32(g)
        else:
            return g

# Explicitly create a fp8 Tensor
class FP8Tensor(torch.Tensor):
    def __new__(cls, data, scale):
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

        return self

    def __repr__(self):
        return f"FP8Tensor(encoded={self._data.view(torch.uint8)}, scale={self._scale}, as_fp32={self.to_fp32()}"

    def to_fp32(self):
        return FP8ConstrFunc.apply(self)

    @classmethod
    def from_fp32(cls, tensor):
        return FP8ConstrFunc.apply(tensor)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is aten.to:
            t, *other_args = args
            assert isinstance(t, FP8Tensor)
            return func(t.to_fp32(), *other_args, **kwargs or {})

        tracker = FP8StatsTracker.get()
        output_scale = tracker.get_next_scale(func)

        # Assume that we have fp8 specific functions for everything!
        fp8_name = f"{torch.overrides.resolve_name(func)}_fp8"
        _, name, overload = fp8_name.split(".")
        try:
            new_func = getattr(getattr(aten, name), overload)
        except Exception as e:
            print(f"No fp8 implementation for {fp8_name}")
            raise e


        new_args = []
        for a in args:
            if isinstance(a, FP8Tensor):
                new_args += [a._data, a._scale]
            else:
                new_args.append(a)

        new_args.append(output_scale)

        # Assumes kwargs never contain Tensors
        fp8_out = new_func(*new_args, **kwargs or {})

        tracker.update_scale(fp8_out)

        return FP8Tensor(fp8_out, output_scale)

# fp8 packing utility for e4m3 format
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

# Register the aten level functions we need.
# These are mostly placeholder and might need to be implemented in c++ as needed
lib = Library("aten", "FRAGMENT")

# Define our new custom functions
lib.define("fp8_from_fp32(Tensor t) -> (Tensor, Tensor)")
lib.define("mm.default_fp8(Tensor m1, Tensor s1, Tensor m2, Tensor s2, Tensor sout) -> Tensor")
# Needed for mm backward
lib.define("t.default_fp8(Tensor t, Tensor s, Tensor sout) -> Tensor")

# Since all my Tensors are on CPU, I register everything there.
def _converter(t):
    # Not sure what this should look like tbh
    # So I just set the scale to 1. for here and do a shady bit packing
    scale = t.new_ones(())
    data = float_to_bits8(t)
    return data, scale
lib.impl("fp8_from_fp32", _converter, "CPU")
def _fp8_mm(m1, s1, m2, s2, sout):
    # Just convert everything to fp32 as I don't have an fp8 kernel at hand
    full_m1 = bits8_to_float(m1) * s1
    full_m2 = bits8_to_float(m2) * s2
    full_out = torch.mm(full_m1, full_m2)
    out = full_out / sout
    return float_to_bits8(out)
lib.impl("mm.default_fp8", _fp8_mm, "CPU")
def _fp8_t(t, s, sout):
    return t.t()
lib.impl("t.default_fp8", _fp8_t, "CPU")

if __name__ == "__main__":
    print("Example of packing/unpacking")
    torch.manual_seed(2)
    x = torch.randn(2) / 5
    print("original", x)
    out = float_to_bits8(x)
    print("fp8 as uint8", out.view(torch.uint8))
    y = bits8_to_float(out)
    print("back to fp32", y)

    NITER = 3

    weights_orig = [torch.randn(2, 2) / 5 for _ in range(NITER)]
    print("weights_orig", weights_orig)
    weights = [FP8Tensor.from_fp32(w) for w in weights_orig]
    print("weights", weights)

    inp_orig = torch.randn(2, 2) / 5

    inp_orig.requires_grad_()

    tmp = inp_orig
    for i in range(NITER):
        tmp = torch.mm(weights_orig[i], tmp)
    print("Result orig:", tmp)
    print("Grad orig:", torch.autograd.grad(tmp.sum(), inp_orig))

    with FP8StatsTracker() as t:
        # Epoc loop
        for i in range(4):
            t.reset()
            inp = FP8Tensor.from_fp32(inp_orig)
            # Model sequential loop
            tmp = inp
            for i in range(NITER):
                tmp = torch.mm(weights[i], tmp)

            print(tmp.requires_grad, tmp.grad_fn)
            print("Result:", tmp)
            print("Grad:", torch.autograd.grad(tmp.to_fp32().sum(), inp_orig))


