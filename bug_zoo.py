import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from base_tensor import BaseTensor
import unittest

class BugZoo(TestCase):
    @unittest.expectedFailure
    def test_binary_ops_swallow_errors(self):
        class BuggyTensor(BaseTensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                raise TypeError("foobar")

        x = BuggyTensor(torch.tensor(1.0))
        self.assertRaisesRegex(TypeError, "foobar", lambda: x + x)


if __name__ == '__main__':
    run_tests()
