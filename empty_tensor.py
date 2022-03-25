import torch
from base_tensor import BaseTensor
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map

from utils import no_dispatch

class EmptyTensor(BaseTensor):
    @staticmethod
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            dtype=elem.dtype, layout=elem.layout, requires_grad=elem.requires_grad,
            device=elem.device
        )

    def __init__(self, elem):
        pass

    def __repr__(self):
        # TODO: this is wrong
        return f'EmptyTensor({self.size()})'

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def inflate(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return torch.ones_like(t, device=t.device)
            else:
                return t

        def deflate(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, cls):
                return EmptyTensor(t)
            else:
                return t

        return tree_map(deflate, super().__torch_dispatch__(
            func, types,
            tree_map(inflate, args),
            tree_map(inflate, kwargs)
        ))


class EmptyTensorTest(TestCase):
    def test_basic(self):
        x = EmptyTensor(torch.randn(4))
        y = EmptyTensor(torch.randn(4))
        r = x + y
        self.assertEqual(r.shape, (4,))


if __name__ == "__main__":
    run_tests()
