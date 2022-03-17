import unittest
import weakref
from collections import defaultdict

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map

alive_tensors = weakref.WeakValueDictionary()

# The main idea behind this tensor is to keep track of what tensors have been
# allocated and track memory allocation through the course of a function.
# Initially inspired by https://github.com/pytorch/pytorch/issues/72450, where I
# wanted to understand *why* PyTorch had a peak memory usage of ~5x the original
# tensor.

name_cnt = defaultdict(int)


class MemoryDebugTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem, func=None):
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem
        if func is not None:
            name = f"{func}_{name_cnt[str(func)]}"
            name_cnt[str(func)] += 1
            alive_tensors[name] = elem
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"MemoryDebugTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"MemoryDebugTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, MemoryDebugTensor) else e

        outs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if e is None:
                return torch.empty(())
            return MemoryDebugTensor(e, func) if isinstance(e, torch.Tensor) else e

        outs = tree_map(wrap, outs)
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        torch.cuda.synchronize()
        print(func)
        print(f"Cur Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        alive_items = [(k, v) for k, v in alive_tensors.items()]
        deduped_tensors = []
        for k, v in alive_items:
            exists_already = False
            for k2, v2 in deduped_tensors:
                if v.storage().data_ptr() == v2.storage().data_ptr():
                    exists_already = True
                    break
            if exists_already:
                continue
            deduped_tensors.append((k, v))
        deduped_tensors = sorted(
            deduped_tensors, key=lambda x: -x[1].storage().nbytes()
        )

        print(
            "Alive Tensors: ",
            [(k, v.storage().nbytes() / 1e9) for k, v in deduped_tensors],
        )
        print()
        return outs


class NegativeTensorTest(TestCase):
    @unittest.skipIf(not TEST_CUDA, "needs cuda")
    def test_construction(self):
        a = MemoryDebugTensor(
            torch.randn(2**27, requires_grad=True, device="cuda"), func="original"
        )
        b = a * 2
        c = a * 4
        self.assertEqual(len(tuple(alive_tensors.keys())), 3)
        del c
        self.assertEqual(len(tuple(alive_tensors.keys())), 2)


if __name__ == "__main__":
    run_tests()
