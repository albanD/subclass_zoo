import torch
from torch.utils._pytree import _register_pytree_node
from torch.fx.experimental.proxy_tensor import make_fx
import torch.fx
from torch.fx import subgraph_rewriter
from torch.overrides import wrap_torch_function

# We use a QTensor class to conveniently pass around both int8 tensor
# as well as scale and zero point necessary to quantize/dequantize them
# when we do graph transformations.

# TODO: Name this something more specific like PerTensorAffineInt8QuantizedTensor
class QTensor:
    tensor: torch.Tensor
    # NB: you could represent these as scalar tensors if you need to
    # trace through them
    scale: float
    zero_point: int

    def __init__(self, tensor, scale, zero_point):
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point

    # NB: wrap_torch_function so that this "factory" function can be
    # symbolically traced as is.  This is not strictly necessary but
    # makes constructing patterns more convenient.
    @staticmethod
    @wrap_torch_function(lambda t, x, y: (t, ))
    def quantize(t: torch.Tensor, scale: float, zero_point: int):
        i8_min = torch.iinfo(torch.int8).min
        i8_max = torch.iinfo(torch.int8).max
        # This formula is probably not quite right, fix it as necessary
        return QTensor(
            torch.clamp(torch.round(t / scale).to(torch.int64) + zero_point, i8_min, i8_max).to(torch.int8),
            scale,
            zero_point
        )

    def dequantize(self):
        return (self.tensor.to(torch.int64) - self.zero_point) * self.scale

# We register it as a pytree node, as in the final graph we want QTensor
# to be eliminated completely (aka QTensor is an entirely out of core concept)
# TODO: We probably could have made QTensor a named tuple and then wouldn't
# need explicit flatten/unflatten

def _qtensor_flatten(q):
    return [q.tensor, q.scale, q.zero_point], None

def _qtensor_unflatten(values, context):
    return QTensor(*values)

_register_pytree_node(QTensor, _qtensor_flatten, _qtensor_unflatten)

# Let's take a simple model that runs linear twice

def f(inp, linear_weight):
    r = torch.nn.functional.linear(inp, linear_weight)
    return torch.nn.functional.linear(r, linear_weight)

# We use the pattern matching API to look for occurrences of linear.

# We use make_fx to generate the sequence of ATen ops that correspond to a
# linear call.  Note that this pattern is only valid if there aren't any
# conditions on, e.g., the shapes of the input tensor.  In general you
# may need a pattern for every permutation of how a composite operator may
# lower; you can get all of them by running through a sufficiently large
# number of example inputs.
# TODO: use symbolic shapes here; this would give you a series of guards
# that would tell you what input sizes the pattern is valid for.
linear_pattern = make_fx(lambda i, w: torch.nn.functional.linear(i, w))(torch.randn(0, 0), torch.randn(0, 0))

# In reality we would first insert observers, and then actually
# insert quantize/dequantize nodes.  In this PoC, I skip observers
# and go straight to quantize/dequantize, and make up random crap for
# the observed quantities.
def linear_replace_fn(i, w):
    fp_i = i.dequantize()
    fp_w = w.dequantize()
    fp_r = torch.nn.functional.linear(fp_i, fp_w)
    # TODO: get the scale and zero_point from observer
    return QTensor.quantize(fp_r, 5.0, 0)
linear_replace = torch.fx.symbolic_trace(linear_replace_fn)

# We first trace out the ATen OP IR of the original model
inp = torch.randn(3, 4)
weight = torch.randn(4, 4)
g = make_fx(f)(inp, weight)
print(g)

# Now, we replace occurrences of linear with quantize/dequantize
subgraph_rewriter.replace_pattern(g, linear_pattern, linear_replace)
print(g)

# Finally, we retrace the model to get lowered operations in terms
# of only pure PyTorch operations.
# TODO: use an interpreter here to preserve stack traces
g2 = make_fx(g)(QTensor(inp, 5.0, 0), QTensor(weight, 5.0, 0))
print(g2)
