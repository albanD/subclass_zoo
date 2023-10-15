import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_map_only
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)

# Subclasses are not very compositional: there is no one true way to
# combine two distinct subclasses into a single one combining both
# of their functionalities.
#
# This file shows a recipe for how to combine a custom parameter subclass
# with a traditional tensor subclass, from Natalia Gimelshein.


# First, the custom parameter subclass is just a subclass of nn.Parameter
# that does NOT make use of the __torch_dispatch__ mechanism.  Typical
# use cases are to annotate parameters with extra methods and data describing
# information about a Parameter that aren't supported on base parameter
# (e.g., sharding.)  Other than that it doesn't integrate with PyTorch
# in any nontrivial way (if it did, we wouldn't be able to combine it.)
class MyParameter(nn.Parameter):
    # This is added to make things work, come back here later
    def __new__(cls, data):
        if isinstance(data, ModeTensor):
            return ModeParameter(data.elem, data.mode)
        return super().__new__(cls, data)

    def custom_fn(self):
        print("Some custom function")


# This is the tensor subclass we want to support.  We've written it in the
# same style as FakeTensor, which also supports a FakeTensorMode which can
# be used to automatically cause plain tensors to be transformed into
# ModeTensors.  In this particular implementation, you can only work with
# ModeTensor inside the mode, but it's also possible to add a
# __torch_dispatch__ implementation that automatically installs the mode
# when a ModeTensor is used without an active mode.
#
# This subclass is written in wrapper tensor style, so elem is probably
# some real tensor.
class ModeTensor(torch.Tensor):
    def __new__(cls, elem, mode):
        res = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size=elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )

        res.elem = elem
        res.mode = mode
        return res

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError("Shouldn't be here")

# The mode is pretty trivial, just wrapping/unwrapping.
class Mode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        def unwrap(e):
            if isinstance(e, ModeTensor):
                return e.elem
            else:
                return e

        def wrap(t):
            if isinstance(t, torch.Tensor):
                return ModeTensor(t, self)
            else:
                return t

        return wrap(func(*tuple(unwrap(a) for a in args), **kwargs))

# So, the key to making this all work, is:
#
# 1. You need to make another class that multiply inherits from ModeTensor
#    and MyParameter.  Order matters as you want to preferentially
#    use ModeTensor to handle methods.
#
# 2. You need to update __new__ on MyParameter to redirect to this class
#    (above) when you get a ModeTensor as argument, so that
#    Parameter(mode_tensor) works.
#
# If your ModeTensor has non-trivial extra data, you have to send all of
# that data to the ModeParameter constructor
class ModeParameter(ModeTensor, MyParameter):
    pass


# See it in action:
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_param = MyParameter(torch.randn(3, 4))

# This works without mode tensor
mod = MyModule()
mod.my_param.custom_fn()

# Now you get a mode tensor
with Mode():
    mod = MyModule()
    print(type(mod.my_param))
    mod.my_param.custom_fn()
