import torch
from utils import no_dispatch
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import run_tests, TestCase

CALLED = []


class ProgressiveLoweringTensor(torch.Tensor):
    @classmethod
    def wrap(cls, t):
        if isinstance(t, torch.Tensor) and not isinstance(t, cls):
            return cls(t)
        else:
            return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.Tensor.relu:
            CALLED.append(func)
            with torch._C.DisableTorchFunction():
                with no_dispatch():
                    return tree_map(cls.wrap, func(*args, **kwargs))
        else:
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        CALLED.append(func)
        with no_dispatch():
            return tree_map(cls.wrap, func(*args, **kwargs))


class ProgressiveLoweringTensorTest(TestCase):
    def test_basic(self):
        CALLED.clear()
        x = ProgressiveLoweringTensor(torch.randn(2))
        x.add(2).relu()
        # add call is low level aten op; relu call is high level torch
        # op
        self.assertEqual(CALLED, [torch.ops.aten.add.Tensor, torch.Tensor.relu])


if __name__ == "__main__":
    run_tests()
