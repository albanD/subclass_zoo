import torch
from torch import Tensor
from torch.utils._pytree import tree_map


# This is a "trivial" wrapper tensor because instead of
# being a tensor (is-a relationship), it instead contains a tensor
# (has-a relationship).  It's trivial because we just directly
# forward all operations onto the inner contained tensor.
# Although this sounds trivial, it is nontrivial, both in terms
# of behavior as well as implementation.
#
#   - Behaviorally, trivial wrapper tensors are complicated because
#     they allow you to layer preexisting tensor features multiple
#     times (ala functorch) in a way that is impossible in normal
#     tensors.  This is because there are two tensors involved:
#     the outer wrapper tensor, as well as the inner tensor.
#
#   - Implementation, trivial wrapper tensors are complicated because
#     the outer wrapper tensor must faithfully replicate all of the
#     properties (including storage and strides) of the inner tensor.
#     This is not so easy to do, and most existing wrapper tensor
#     implementations in the wild do not do this correctly, and
#     subsequently fail asserts in PyTorch autograd when running
#     PyTorch with DEBUG.
#
# This tensor could have been implemented in terms of Alban's
# WrapperTensor, but I wanted to keep all of the implementation
# in one place for easier modification, because as you will see,
# doing this completely correctly is quite involved.
class TrivialWrapperTensor(Tensor):
    # At the moment, report meta device only

    # We have an interesting problem for the constructor.  What if you
    # pass in a view to the TrivialWrapperTensor?  Do we accurately
    # represent the storage in this situation.  If we accurately represent it,
    # then what if you call TrivialWrapperTensor on that view again; there
    # is no way to recover the new meta storage you had previously allocated.
    # If we don't accurately represent it, we're at risk of failing
    # autograd tests (but maybe this is OK if you don't expect to
    # autograd across the boundary).
    #
    # How to autograd through the constructor of TrivialWrapperTensor?
    @staticmethod
    def __new__(cls):
        pass

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def to_elem(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        def to_meta(t):
            if isinstance(t, cls):
                return torch.Tensor._make_subclass(torch.Tensor, t)
            else:
                return t

        r_meta = func(*tree_map(to_meta, args), **tree_map(to_meta, kwargs))
        r = Tensor._make_subclass(cls, r_meta)
        r.elem = func(*tree_map(to_elem, args), **tree_map(to_elem, kwargs))

        return r
