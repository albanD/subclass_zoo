import torch
from torch import Tensor
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import (
    TestCase, run_tests, disable_gc
)

import weakref


# This is a "trivial" wrapper tensor because instead of being a tensor (is-a
# relationship), it instead contains a tensor (has-a relationship).  It's
# trivial because we just directly forward all operations onto the inner
# contained tensor.
#
# This wrapper is implemented in a very funny way: it assumes the inner
# metadata is exactly synchronized with the outer metadata; in particular, the
# storage of the outer wrapper points to the storage of the inner wrapped
# tensor!  So you might wonder what the point of this wrapper tensor is, since
# the inner and outer tensor look exactly the same.  Well:
#
#   - The inner and outer tensors can have differing metadata for other
#     subsystems; e.g., you can run autograd multiple times using this wrapper.
#
#   - You don't have to use no_dispatch() to get to the "inner" layer, you
#     just unwrap and go.
#
# Sometimes some metadata needs to differ between inner and outer, and that
# gets complicated.  Coming soon!
class TrivialWrapperTensor(Tensor):
    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem):
        # See Note [Passing requires_grad=true tensors to subclasses]
        assert not elem.requires_grad or not torch.is_grad_enabled()
        return Tensor._make_subclass(cls, elem)

    def __init__(self, elem):
        self.elem = elem

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        # You don't have to wrap; without wrapping you lose the wrapper
        # tensor after any operation, which may be desirable for some
        # use cases
        def wrap(t):
            if isinstance(t, Tensor):
                return TrivialWrapperTensor(t)
            else:
                return t

        return tree_map(
            wrap,
            func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        )


TWT = TrivialWrapperTensor


class TrivialWrapperTensorTest(TestCase):
    def test_no_cycle(self):
        fins = []
        with disable_gc():
            r = TWT(torch.empty(2))
            w = weakref.ref(r, lambda _: fins.append(1))
            self.assertEqual(fins, [])
            del r
            self.assertEqual(fins, [1])
            del w

    def test_no_copy(self):
        inner = torch.empty(2)
        outer = TWT(inner)
        self.assertEqual(inner.data_ptr(), outer.data_ptr())

    def test_basic(self):
        self.assertEqual(
            TWT(torch.tensor(1.0)) + TWT(torch.tensor(2.0)),
            TWT(torch.tensor(3.0)))


if __name__ == '__main__':
    run_tests()


# Random commentary
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
#
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
#
# Current idea:
#   - constructor is OK, even for views, but we'll construct a fresh
#     storage on entry each time.  use_count 1 on storage is safest
#     but if you wrap the same tensor multiple times they are
#     disconnected
#
# Another idea for storage is to point to the SAME storage as the
# tensor we're wrapping
