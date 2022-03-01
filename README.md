# subclass zoo

This repository contains a number of examples of Tensor subclasses in PyTorch,
specifically using `__torch_dispatch__` to integrate deeply into PyTorch's
existing subsystems.  We're still working out good APIs for working with
Tensor subclasses, and this repository is here to tell you about what we've
figured out so far!

Here's what's in the repo so far:

* `negative_tensor.py` - a reimplementation of negative tensor views as
  implemented in PyTorch core (https://github.com/pytorch/pytorch/pull/56058)
