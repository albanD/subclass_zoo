import torch
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests

import weakref
alive_tensors = weakref.WeakValueDictionary()

# The main idea behind this tensor is to keep track of what tensors have been
# allocated and track memory allocation through the course of a function.
# Initially inspired by https://github.com/pytorch/pytorch/issues/72450, where I
# wanted to understand *why* PyTorch had a peak memory usage of ~5x the original
# tensor.

class MemoryDebugTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, func=None):
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        r.elem = elem
        if func is not None:
            idx = 0
            name = f'{func}'
            while name in alive_tensors:
                idx += 1
                name = f'{func}_{idx}'
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
        # no_dispatch is only needed if you use enable_python_mode.
        # It prevents infinite recursion.
        print(func)
        outs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))

        def wrap(e):
            if e is None:
                return torch.empty(())
            return MemoryDebugTensor(e, func) if isinstance(e, torch.Tensor) else e

        outs = tree_map(wrap, outs)
        # torch.cuda.synchronize()
        # import gc; gc.collect()
        # torch.cuda.empty_cache()
        print(f"Cur Memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB")
        alive_items = [(k, v) for k,v in alive_tensors.items()]
        deduped_tensors = []
        for k, v in alive_items:
            exists_already = False
            for k2, v2 in deduped_tensors:
                if v.data_ptr() == v2.data_ptr():
                    exists_already = True
                    break
            if exists_already:
                continue
            deduped_tensors.append((k, v))
        deduped_tensors = sorted(deduped_tensors, key=lambda x: -x[1].storage().nbytes())

        print("Alive Tensors: ", [(k, v.storage().nbytes()/2**30) for k, v in deduped_tensors])
        print()
        return outs


a = MemoryDebugTensor(torch.randn(2**25, requires_grad=True, device='cuda'), func="original")
torch.softmax(a, dim=0).sum().backward()
