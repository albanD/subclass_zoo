import weakref

import torch

from base_tensor import BaseTensor
from torch import Tensor
from torch.testing._internal.common_utils import (
    disable_gc,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.utils._pytree import tree_map

# In a lot of use cases for tensor subclasses, there is a concept
# of an "inner" tensor, which is a normal, non-subclassed tensor
# that after you do your stuff you can redispatch to.  This file gives recipes
# for a number of trivial tensors; tensors which look and behave exactly like
# their inner tensors, and propagate themselves through all invocations.  As
# it turns out, there are a number of different ways to do the same thing.
# However, the main axis of variation is this:
#
#       Do you actually store the inner tensor (composition / has-a
#       relationship) or do you make what is effectively a super call
#       (inheritance / is-a relationship)
#
# These options have different tradeoffs which are discussed in more
# detail below.  Hopefully this file will give you an idea about some of the
# tools in your toolbox.
#
# WARNING: These tensors inherit from BaseTensor, which is a local
# compatibility shim that tracks changes to Tensor that we intend to make but
# haven't made it to core.  If you want to use these classes you will need to
# include the extra bits from BaseTensor.
#
# TODO: Channeling Alban, we probably want to take some of these exemplars and
# turn them into part of the official public API, so end users don't have to
# copy paste them into their own functions.
#
# TODO: Redo these examples with compositionality in mind.


class TrivialTensorViaInheritance(BaseTensor):
    """
    TrivialTensorViaInheritance extends tensor behavior using inheritance ("is
    a").  These implementations are very straightforward and we recommend
    using them if it works for your use case.  To get the base behavior,
    you use standard object-oriented idiom of super().

    Benefits and downsides of this representation:

        + Efficient representation (only one tensor).
        + Do not have to worry about synchronizing metadata between the inner
          and outer tensor.
        = Requires multiple inheritance to do composition.  This *does*
          work, but it is a bit mind-bending, you have to deal with the
          diamond inheritance problem, and traditionally you only get a fixed
          set of composition (rather than dynamic, as in functorch) unless
          you're willing to generate classes on the fly.
        - Doesn't work if you need to run internal PyTorch subsystems
          (e.g., autograd) multiple times.
        - Doesn't work if the internal tensor has a different shape
          than the outer tensor.
        - Doesn't work if you need multiple internal tensors.
    """

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def wrap(t):
            # When could the returned tensor already be our subclass?
            # The most common situation is when an input tensor
            # is returned as an output tensor, e.g., inplace or out
            # implementations.
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, super().__torch_dispatch__(func, types, args, kwargs))


class TrivialTensorViaComposition(BaseTensor):
    """
    TrivialTensorViaComposition extends tesor behavior using composition ("has
    a").  You can see that unlike inheritance, we save the original tensor in
    a field in the tensor.  These are often referred to as "wrapper tensors",
    as you are wrapping the original tensor.

    The nuance of wrapper tensors is that the external wrapper tensor is still
    required to have all of the metadata that the inner tensor has; this
    includes stride and storage!  In this example, we assume the inner and
    outer metadata is exactly synchronized... so in fact the wrapper tensor is
    literally just a TrivialTensorViaInheritance (in particular, the outer
    wrapper points to the same storage as the inner wrapped tensor).  The only
    difference is that we've also recorded the original tensor as an element
    on the class as well.

    Benefits and downsides of this representation:

    + Many people find perform operations in the inner layer more
      intuitive (just unwrap the tensor)
    + In principle, is compositional with other tensor subclasses; in
      practice, compositionality in this way is hard to understand
      without more structure (e.g., functorch)
    + Allows you to use PyTorch's subsystems (e.g., autograd) multiple
      times (e.g., as done in functorch)
    + Metadata between the inside and outside can diverge (not shown in
      this example, TODO: add to zoo)
    - Representation requires two tensors (inner and outer); sometimes
      this is unnecessary
    - You must synchronize the metadata for the two tensors.  Historically
      we had a number of incomplete/incorrect implementations of this;
      this file shows how to correctly (and easily).
    """

    def __init__(self, elem):
        super().__init__(elem)
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t)
            else:
                return t

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))


parametrize_trivial = parametrize(
    "TrivialTensor",
    [
        TrivialTensorViaInheritance,
        TrivialTensorViaComposition,
    ],
    name_fn=lambda x: x.__name__,
)


# We run our tests on both formulations of trivial tensors to show that
# in the trivial case, they are exactly equivalent
class TrivialTensorTest(TestCase):
    @parametrize_trivial
    def test_no_cycle(self, TrivialTensor):
        fins = []
        with disable_gc():
            r = TrivialTensor(torch.empty(2))
            w = weakref.ref(r, lambda _: fins.append(1))
            self.assertEqual(fins, [])
            del r
            self.assertEqual(fins, [1])
            del w

    @parametrize_trivial
    def test_no_copy(self, TrivialTensor):
        inner = torch.empty(2)
        outer = TrivialTensor(inner)
        self.assertEqual(inner.data_ptr(), outer.data_ptr())

    @parametrize_trivial
    def test_basic(self, TrivialTensor):
        # NB: this is not so basic, this executes a shit ton of
        # ops, including inplace ops
        self.assertEqual(
            (TrivialTensor(torch.tensor(1.0)) + TrivialTensor(torch.tensor(2.0))),
            TrivialTensor(torch.tensor(3.0)),
        )


instantiate_parametrized_tests(TrivialTensorTest)


if __name__ == "__main__":
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
