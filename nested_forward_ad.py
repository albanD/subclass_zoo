import torch
from torch.autograd import forward_ad as fwAD
from torch import Tensor
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import (
    TestCase, run_tests, disable_gc, parametrize,
    instantiate_parametrized_tests
)
from torch.overrides import enable_reentrant_dispatch

import functools
import contextlib

from utils import no_dispatch
from base_tensor import BaseTensor

# WARNING:
# This class requires https://github.com/pytorch/pytorch/pull/73925 (that was reverted)
# to properly work with forward AD implementation
# If you get an error about "Expected this function to only be reached in inference mode"
# then you don't have that patch!

# This class wraps a pytorch dual Tensor and associates a level to it.
# This allows to do multi-level forward AD even though pytorch only
# support one level.
class ForwardADTensor(BaseTensor):
    @staticmethod
    def __new__(cls, dual_t, *, level, ignore_no_grad=False):
        # Use this to check if the plain object has a forward grad or not while ignoring
        # all of the torch_dispatch handling
        with no_dispatch():
            primal, tangent = fwAD.unpack_dual(dual_t)
        # Ensure we actually have a dual Tensor
        assert ignore_no_grad or tangent is not None, "ForwardADTensor can only wrap Tensors with forward gradients"
        # Ensure that nesting is happening in the right order
        if isinstance(dual_t, cls):
            assert dual_t.level < level, "Level ordering is wrong!"
        res = super().__new__(cls, primal)
        return res

    def __repr__(self):
        # Use no_dispatch here to get a plain representation of this Tensor without any of the
        # torch_dispatch handling
        with no_dispatch():
            self_repr = super().__repr__()
        return f"ForwardADTensor{self.level}({self_repr}, {self.elem!r})"

    def __init__(self, dual_t, *, level, ignore_no_grad=False):
        self.elem = dual_t
        self.level = level

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Detach is a special case here.
        # This special case is for the code from autograd that uses shallow_copy_and_detach
        # (which is rerouted to detach in torch_dispatch) and user code that calls detach
        # In this case, we want to only detach *one* level of forward grad. Since forward grad
        # is already handled before getting here, we just want to convert detach into alias before
        # applying it to the underlying Tensor.
        # We also need a special case to force wrapping even though there aren't any forward grad (yet)
        # as the forward grad will be associated to the result in the dispatcher on the return from this
        # call.
        ignore_no_grad = False
        if func is torch.ops.aten.detach.default:
            ignore_no_grad = True
            func = torch.ops.aten.alias.default

        max_level = -1

        def find_level(t):
            nonlocal max_level
            if isinstance(t, cls):
                max_level = max(max_level, t.level)

        # TODO: don't use tree_map
        tree_map(find_level, args)
        tree_map(find_level, kwargs)


        def matches_level(t):
            return isinstance(t, cls) and t.level == max_level

        def unwrap(t):
            # All the Tensors at this level must be unpacked so that the new call into the
            # dispatcher will handle this level of forward AD
            if matches_level(t):
                return t.elem
            else:
                # If we get a forward AD Tensor here, its level have been handled in the dispatcher
                # call that lead to this torch dispatch. So now we want to just consider it as a
                # constant for level during the next call into the dispatcher.
                if isinstance(t, torch.Tensor) and fwAD.unpack_dual(t).tangent is not None:
                    return fwAD.unpack_dual(t).primal
                return t

        def wrap(t):
            if isinstance(t, Tensor) and not matches_level(t):
                # Only wrap Tensors that have a tangent
                # or are about to get one (when calling detach)
                tp, td = fwAD.unpack_dual(t)
                if td is not None or ignore_no_grad:
                    return cls(t, level=max_level, ignore_no_grad=ignore_no_grad)
            return t

        with enable_reentrant_dispatch():
            return tree_map(
                wrap,
                func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            )

class NestedForwardADTest(TestCase):
    def test_basic(self):
        # We could have a better UX for making sure forward AD is enabled.
        # For simplicity here, we just keep it enabled for all the test
        with fwAD.dual_level():
            t_p = torch.rand(2)
            t_t = torch.rand(2)
            t = ForwardADTensor(fwAD.make_dual(t_p, t_t), level=0)
            out = t * 2
            out_p, out_t = fwAD.unpack_dual(out.elem)
            self.assertEqual(out_p, 2 * t_p)
            self.assertEqual(out_t, 2 * t_t)

    def test_nested(self):
        with fwAD.dual_level():
            t_p = torch.rand(2)
            t_t = torch.rand(2)
            t = ForwardADTensor(fwAD.make_dual(t_p, t_t), level=1)

            t2_t = torch.rand(2)
            # There is only one order of nesting that makes sense!
            with self.assertRaisesRegex(AssertionError, "Level ordering is wrong!"):
                t2 = ForwardADTensor(fwAD.make_dual(t, t2_t), level=0)

            # Note that both gradients are on the primal. So we do *not* compute
            # higher order derivatives here!
            t2 = ForwardADTensor(fwAD.make_dual(t, t2_t), level=2)

            # Make sure that t2 has all the right metadata
            self.assertIsInstance(t2, ForwardADTensor)
            self.assertEqual(t2.level, 2)
            self.assertEqual(t2, t_p)
            self.assertIsNone(fwAD.unpack_dual(t2).tangent)
            elem = t2.elem
            self.assertIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem.level, 1)
            self.assertEqual(elem, t_p)
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t2_t)
            elem = elem.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_p)
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t_t)

            # Simple op that doesn't take extra arguments
            out = t2.exp()

            # Make sure that ops of t2 compute both levels of autograd independently
            self.assertIsInstance(out, ForwardADTensor)
            self.assertEqual(out.level, 2)
            self.assertEqual(out, t_p.exp())
            self.assertIsNone(fwAD.unpack_dual(out).tangent)
            elem = out.elem
            self.assertIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem.level, 1)
            self.assertEqual(elem, t_p.exp())
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t2_t * t_p.exp())
            elem = elem.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_p.exp())
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t_t * t_p.exp())

            # Computing higher order derivatives now!
            t = ForwardADTensor(fwAD.make_dual(t_t, t2_t), level=1)
            t2 = ForwardADTensor(fwAD.make_dual(t_p, t), level=2)

            # Make sure that t2 has all the right metadata
            self.assertIsInstance(t2, ForwardADTensor)
            self.assertEqual(t2.level, 2)
            self.assertEqual(t2, t_p)
            self.assertIsNone(fwAD.unpack_dual(t2).tangent)
            elem = t2.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_p)
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t_t)
            elem = fwAD.unpack_dual(elem).tangent
            self.assertIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem.level, 1)
            self.assertEqual(elem, t_t)
            self.assertIsNone(fwAD.unpack_dual(elem).tangent)
            elem = elem.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_t)
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t2_t)

            # An op with different first and second derivative
            out = t2.pow(2)

            # Make sure that ops of t2 computes higher order derivatives
            self.assertIsInstance(out, ForwardADTensor)
            self.assertEqual(out.level, 2)
            self.assertEqual(out, t_p.pow(2))
            self.assertIsNone(fwAD.unpack_dual(out).tangent)
            elem = out.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_p.pow(2))
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t_t * 2 * t_p)
            elem = fwAD.unpack_dual(elem).tangent
            self.assertIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem.level, 1)
            self.assertEqual(elem, t_t * 2 * t_p)
            self.assertIsNone(fwAD.unpack_dual(elem).tangent)
            elem = elem.elem
            self.assertNotIsInstance(elem, ForwardADTensor)
            self.assertEqual(elem, t_t * 2 * t_p)
            self.assertEqual(fwAD.unpack_dual(elem).tangent, t2_t * 2 * t_p)


    def test_no_confusion(self):
        # This test ensure that we don't do "perturbation confusion"
        # meaning that gradients at each levels are indeed computed independently
        # and don't interact with each other
        with fwAD.dual_level():
            t_p = torch.rand(2)
            t_t = torch.rand(2)
            t = ForwardADTensor(fwAD.make_dual(t_p, t_t), level=0)
            t2_p = torch.rand(2)
            t2_t = torch.rand(2)
            t2 = ForwardADTensor(fwAD.make_dual(t2_p, t2_t), level=1)

            mixed_out = t * t2

            mixed_out_lvl1_p, mixed_out_lvl1_t = fwAD.unpack_dual(mixed_out.elem)
            mixed_out_lvl0_p, mixed_out_lvl0_t = fwAD.unpack_dual(mixed_out.elem.elem)
            self.assertEqual(mixed_out_lvl1_p, t_p * t2_p)
            self.assertEqual(mixed_out_lvl1_t, t2_t * t_p)
            self.assertEqual(mixed_out_lvl0_p, t_p * t2_p)
            self.assertEqual(mixed_out_lvl0_t, t_t * t2_p)



if __name__ == '__main__':
    run_tests()



