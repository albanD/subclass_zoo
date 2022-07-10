# subclass zoo

This repository contains a number of examples of Tensor subclasses in PyTorch,
specifically using `__torch_dispatch__` to integrate deeply into PyTorch's
existing subsystems (there's also some use of modes as well).  We're still
working out good APIs for working with Tensor subclasses, and this repository
is here to tell you about what we've figured out so far!  To run these
examples, you will want a recent nightly of PyTorch.

Here's what's in the repo so far:

- `inner_autograd_tensor.py` shows how to override autograd from
  `__torch_dispatch__`, by deferring autograd to the inner tensor on a
  subclass.
- `negative_tensor.py` is a reimplementation of negative tensor views as
  implemented in PyTorch core (https://github.com/pytorch/pytorch/pull/56058)
- `python_meta_tensor.py` is a demonstration of how to extend an existing
  tensor (meta tensor) with some extra behavior (in this case, implementations
  of meta functions for operations that don't support it natively)
- `sparse_output.py`
- `tracer_tensor.py`
- `trivial_tensors.py` is a comparison for two ways how to "wrap" tensors,
  one using inheritance (is-a) and one using composition (has-a) (so called
  wrapper tensors)
- `verifier_tensor.py`

There are also some utility files:

- `base_tensor.py` contains a common superclass that most of our tensors
  inherit from, that fixes up some problems with directly inheriting from
  torch.Tensor.  We intend to upstream these changes so that this superclass
  is not necessary.
- `utils.py` contains some handy utility functions that we found ourselves
  repeatedly using in our implementations.

We're still working on the APIs in questions, so sometimes there will be bugs.
`bug_zoo.py` contains repros for known bugs we're tracking in PyTorch proper.

TODO

- CUDA sanitizer in Python (hard cuz no event hooks)
- Sparse gradients / outputs per Christian (using modes; gradients hard cuz
  need torch function mode)
- SSD tensor
- Reimplement functionalization tensor
- Nested tensor
- Custom allocator mode (albanD)
- Lazy tensor
- Immutable tensor
- Various ways of writing FX passes https://gist.github.com/1c640ea30fd7451b08e90e34461459c1

## Work plan

* TODO: merge BaseTensor into Tensor
  * TODO: torch function disable https://github.com/pytorch/pytorch/pull/73942
* Get rid of `fill_defaults`

* Compositionality
  * TODO: suppress elem in init

## Developer notes

* This repo is formatted with ufmt and autoflakes.  Use `./format.sh` to
  reformat all the files in this repository.
