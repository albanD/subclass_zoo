import unittest

import torch

from base_tensor import BaseTensor
from trivial_tensors import TrivialTensorViaInheritance
from torch.testing._internal.common_utils import run_tests, TestCase


class BugZoo(TestCase):
    @unittest.expectedFailure
    def test_binary_ops_swallow_errors(self):
        class BuggyTensor(BaseTensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise TypeError("foobar")

        x = BuggyTensor(torch.tensor(1.0))
        self.assertRaisesRegex(TypeError, "foobar", lambda: x + x)

    @unittest.skip
    def test_super_dispatch_segfault(self):
        class SuperDispatchSegfaultTensor(BaseTensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return super().__torch_dispatch__(func, types, list(args), kwargs)

        SuperDispatchSegfaultTensor(torch.tensor(1.0)).neg()

    # Fixed!
    def test_trivial_inplace(self):
        x = TrivialTensorViaInheritance(torch.tensor(1.0))
        y = x * torch.tensor(1.0, requires_grad=True)
        y.relu_()

    # Fixed!
    def test_grad_fn(self):
        class TestTensor(BaseTensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                if func is torch.ops.aten.add.Tensor and 'alpha' in kwargs:
                    # decompose it
                    r = torch.add(args[0], args[1] * kwargs['alpha'])
                    self.assertIsNone(r.grad_fn)
                    return r
                return super().__torch_dispatch__(func, types, args, kwargs)

        x = TestTensor(torch.tensor(1.0)).requires_grad_()
        y = TestTensor(torch.tensor(2.0)).requires_grad_()
        torch.add(x, y, alpha=2)


if __name__ == "__main__":
    run_tests()
