# subclass zoo

This repository contains a number of examples of Tensor subclasses in PyTorch,
specifically using `__torch_dispatch__` to integrate deeply into PyTorch's
existing subsystems.  We're still working out good APIs for working with
Tensor subclasses, and this repository is here to tell you about what we've
figured out so far!

Here's what's in the repo so far:

- `negative_tensor.py` - a reimplementation of negative tensor views as
  implemented in PyTorch core (https://github.com/pytorch/pytorch/pull/56058)
- `python_meta_tensor.py` - a demonstration of how to extend an existing
  tensor (meta tensor) with some extra behavior (in this case, implementations
  of meta functions for operations that don't support it natively)
- `trivial_tensors.py` - a comparison for two ways how to "wrap" tensors,
  one using inheritance (is-a) and one using composition (has-a) (so called
  wrapper tensors)

TODO

- Implement speculate and validate tensor
  https://docs.google.com/document/d/1s44XJg_AQFZbSm4X1u9bE6V2ELstFdlekA_m2lNWn1Y/edit#heading=h.5pt6253mfam1
- CUDA sanitizer in Python (hard cuz no event hooks)
- Sparse gradients / outputs per Christian (using modes)
- SSD tensor
- Reimplement functionalization tensor
- Nested tensor
