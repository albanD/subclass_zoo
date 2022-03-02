import torch
from torch import Tensor
from torch.autograd import Function
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests

from utils import no_dispatch

import unittest


# This is a reimplementation of "negative tensor views" as currently
# implemented in PyTorch core.  This lets you represent a negation
# without actually materializing the negated value, so it can be fused
# with further operations.  See also the PR that added this to PyTorch:
# https://github.com/pytorch/pytorch/pull/56058
class NegativeTensor(Tensor):
    @staticmethod
    def __new__(cls, elem):
        # At the moment, this class is not compositional, so we assert
        # that the tensor we're wrapping is exactly a Tensor
        assert type(elem) is Tensor

        # Note [Passing requires_grad=true tensors to subclasses]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Calling _make_subclass directly in an autograd context is
        # never the right thing to do, as this will detach you from
        # the autograd graph.  You must create an autograd function
        # representing the "constructor" (NegativeView, in this case)
        # and call that instead.  This assert helps prevent direct usage
        # (which is bad!)
        assert not elem.requires_grad or not torch.is_grad_enabled()

        # There is something very subtle going on here.  In particular,
        # suppose that elem is a view.  Does all of the view metadata
        # (sizes, strides, storages) get propagated correctly?  Yes!
        # Internally, the way _make_subclass works is it creates an
        # alias (using Tensor.alias) of the original tensor, which
        # means we replicate storage/strides, but with the Python object
        # as an instance of your subclass.  In other words,
        # _make_subclass is the "easy" case of metadata propagation,
        # because anything that alias() propagates, you will get in
        # your subclass.  It is _make_wrapper_subclass which is
        # problematic...
        #
        # TODO: We need to think about how we want to turn this into
        # official API.  I am thinking that something that does the
        # assert above and this call could be made into a utility function
        # that is in the public API
        return Tensor._make_subclass(cls, elem)

    def __repr__(self):
        with no_dispatch():
            return repr(self.neg())

    def physical_repr(self):
        with no_dispatch():
            return f"negative_view({super().__repr__()})"

    # Without this, the default __torch_function__ implementation will
    # attempt to wrap the returned tensor for any operation in a NegativeView
    # (wrong wrong wrong)
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # TODO: inplace and out

        # This implements fallback behavior, where we materialize the
        # negative view into a normal tensor, and then do the operation on
        # normal tensors.  Because we eliminate all negative views before
        # performing our operation, no_dispatch() is not necessary here.
        def unwrap(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return t.neg()
            else:
                return t
        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


# A differentiable function that takes a negative view on a function.  Use
# this to construct NegativeTensors.
class NegativeView(Function):
    @staticmethod
    def forward(ctx, input):
        # Exact type matches as NegativeTensor is not compositional
        if type(input) is NegativeTensor:
            # If we are passed a NegativeTensor, we can simply alias it as
            # a normal tensor and return it.
            # TODO: this should be in standard library
            return torch.Tensor._make_subclass(torch.Tensor, input)
        elif type(input) is Tensor:
            return NegativeTensor(input)
        else:
            raise AssertionError("negative tensors are not yet compositional")

    @staticmethod
    def backward(ctx, grad):
        return negative_view(grad)


negative_view = NegativeView.apply


class NegativeTensorTest(TestCase):
    def test_construction(self):
        # NegativeTensor is semantically equivalent to negating the tensor
        self.assertEqual(NegativeTensor(torch.tensor(1)), torch.tensor(-1))
        self.assertEqual(negative_view(torch.tensor(1)), torch.tensor(-1))

        # The direct constructor is not valid in autograd contexts; you must
        # use negative_view
        self.assertRaises(
            Exception,
            lambda: NegativeTensor(torch.empty(1, requires_grad=True)))
        self.assertRaises(
            Exception,
            lambda: NegativeTensor(torch.empty(1, requires_grad=True).sum()))
        negative_view(torch.empty(1, requires_grad=True))
        negative_view(torch.empty(1, requires_grad=True).sum())

        # The tensor is aliases with its original
        x = torch.tensor(1.)
        y = negative_view(x)
        self.assertEqual(y, torch.tensor(-1.))
        x.add_(1)
        self.assertEqual(y, torch.tensor(-2.))

    def test_repr(self):
        x = negative_view(torch.tensor(1))

        # I decided to make the normal repr print "as if" it were a normal
        # tensor
        self.assertExpectedInline(repr(x), """tensor(-1)""")

        # physical_repr tells you if something funny is going on
        self.assertExpectedInline(x.physical_repr(), """\
negative_view(tensor(1))""")

    def test_functional(self):
        self.assertEqual(negative_view(torch.tensor(1)) + 1, torch.tensor(0))

    def test_backward(self):
        base = torch.tensor(-1.0, requires_grad=True)
        x = negative_view(base)
        x.sum().backward()
        self.assertEqual(base.grad, torch.tensor(-1.0))

    def test_negative_view_of_view(self):
        base = torch.zeros(2, 2)
        view = base[0]
        neg_view = negative_view(view)
        self.assertEqual(neg_view, torch.zeros(2))
        base[0, 0].add_(1)
        base[0, 1].add_(2)
        base[1, 0].add_(3)
        base[1, 1].add_(4)
        self.assertEqual(neg_view, torch.tensor([-1.0, -2.0]))

    # autograd custom functions with views don't work
    # tracked in https://github.com/pytorch/pytorch/issues/73604
    @unittest.expectedFailure
    def test_view_backward(self):
        base = torch.tensor(1.0, requires_grad=True)
        z = base * 1
        x = negative_view(z)
        z.mul_(-1)
        # Uncomment this line, which manually recomputes the view, to make this
        # test pass while master is broken
        # x = negative_view(z)
        x.sum().backward()
        self.assertEqual(base.grad, torch.tensor(1.0))

    @unittest.expectedFailure
    def test_non_subclass_view_backward(self):
        class Alias(Function):
            @staticmethod
            def forward(ctx, input):
                return input[:]

            @staticmethod
            def backward(ctx, grad):
                return grad

        base = torch.tensor([1.0], requires_grad=True)
        z = base * 1
        x = Alias.apply(z)
        z.mul_(-1)
        x.sum().backward()
        self.assertEqual(base.grad, torch.tensor([-1.0]))


if __name__ == '__main__':
    run_tests()
