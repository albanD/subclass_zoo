import torch
from torch.testing._internal.common_utils import (
    TestCase, run_tests
)
from torch.utils._pytree import tree_map
import torch.nn.functional

from base_tensor import BaseTensor
from utils import fill_defaults

# This file describes how to use wrapper tensors (ala TrivialTensorViaComposition)
# to override autograd from __torch_dispatch__.  Ordinarily,
# __torch_dispatch__ runs after autograd, so you have no way of overriding
# the autograd behavior (since it will be handled after you return).  However,
# if we put the autograd tensor *inside* a wrapper tensor (which doesn't
# itself require gradients), we get a chance to interpose (in __torch_dispatch__)
# before you handle gradients on the inner element.
#
# Note that you can also use __torch_function__ instead to implement this
# functionality, so this is mostly a question of whether or not you want to
# target the public Python API, or the internal ATen operators API
# (torch.ops.aten).

class InnerAutogradTensor(BaseTensor):
    @staticmethod
    def __new__(cls, elem, *, requires_grad=None):
        # Outer tensor's autograd is now disconnected from the inner
        # tensors autograd...
        return super().__new__(cls, elem, requires_grad=False)

    def __init__(self, elem):
        # ... but note that we save the inner tensor, so we can still
        # do autograd on operations on the inside!
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                return t.elem
            else:
                return t

        def wrap(t):
            # Micro-optimization: not necessary to rewrap if the output tensor
            # doesn't require gradients
            if isinstance(t, torch.Tensor) and not isinstance(t, cls) and t.requires_grad:
                return cls(t)
            else:
                return t

        # Override gradient behavior
        if func == torch.ops.aten.embedding.default:
            args = fill_defaults(args, 5, [-1, False, False])
            weight, indices, padding_idx, scale_grad_by_freq, _sparse = map(unwrap, args)
            assert not kwargs
            # Force sparse gradients.  We could have also done this by
            # defining a custom autograd function.
            return cls(func(weight, indices, padding_idx, scale_grad_by_freq, True))

        return tree_map(
            wrap,
            func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
        )

class InnerAutogradTensorTest(TestCase):
    def test_basic(self):
        x = torch.randn(1, requires_grad=True)
        y = InnerAutogradTensor(x)
        self.assertFalse(y.requires_grad)
        self.assertTrue(y.elem.requires_grad)
        z = InnerAutogradTensor(x)
        # Although y and z do not require grad, we are still able
        # to differentiate
        r = y + z
        # Note we have to extract out the inner tensor (which requires_grad)
        # to actually differentiate
        r.sum().elem.backward()
        self.assertEqual(x.grad, torch.tensor([2.0]))  # two uses!

    def test_embedding(self):
        input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        weights = torch.rand(10, 3, requires_grad=True)
        embedding_matrix = InnerAutogradTensor(weights)
        r = torch.nn.functional.embedding(input, embedding_matrix)
        r.sum().elem.backward()
        # Gradient is sparse even though we didn't ask for it in embedding!
        self.assertTrue(weights.grad.is_sparse)

    def test_mixing(self):
        # Mixing behavior is confusing.  Let's take a look
        w1 = torch.randn(1, requires_grad=True)
        w2 = torch.randn(1, requires_grad=True)

        # Autograd doesn't "unwrap" variables, they still remember if they
        # requires_grad; and in fact, inside __torch_function__ it is willing
        # to mix gradients between multiple levels.  This can be very
        # confusing so be careful!
        x = InnerAutogradTensor(w1) + w2
        g1, g2 = torch.autograd.grad(x.elem.sum(), (w1, w2))
        self.assertEqual(g1, torch.ones(1))
        self.assertEqual(g2, torch.ones(1))

        # Hopefully this makes more sense: w1 doesn't require gradients from
        # the outer context (due to InnerAutogradTensor), so it doesn't get
        # a gradient here.  You could make it also have gradients by
        # not detaching from __new__, but once again... very confusing.
        x = InnerAutogradTensor(w1) + w2
        g1, g2 = torch.autograd.grad(x.sum(), (w1, w2), allow_unused=True)
        self.assertEqual(g1, None)
        self.assertEqual(g2, torch.ones(1))

if __name__ == '__main__':
    run_tests()
