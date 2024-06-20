# How to create your own tensors without writing C++

## The old world

TODO: Make the old world story punchier, describe in detail the pain people go through when writing their own tensors and convincing the core team to add a dispatch key.

Creating your own Tensor Types is a fairly natural problem maybe you'd like to support Tensors with an error bound, jagged tensors, diagonal tensors, sparse tensors, tensors with quaternions as their values. Numerical computing experts can get pretty creative yet the tooling hasn't really been user friendly for PyTorch programmers to build these kinds of tensors.

In the good old days if you needed to create your own custom `Tensor` in PyTorch you needed to write A LOT of C++ and also convince someone on the core team to give you one of the 14 available Dispath keys.

These are both hard sells. Writing C++ is tricky and again you don't just need to write a function or two. You basically needed to write the equivalent of https://github.com/pytorch/nestedtensor

```
Mark@DESKTOP-LJR0S1S MINGW64 ~/Dev/nestedtensor (master)
$ cloc .
     124 text files.
     124 unique files.
      30 files ignored.

github.com/AlDanial/cloc v 1.82  T=1.00 s (119.0 files/s, 21271.0 lines/s)
-------------------------------------------------------------------------------
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          44           1232           1087           5699
C++                             27            388            125           4478
CUDA                             6            289            135           2001
C/C++ Header                    22            212            165           1781
Jupyter Notebook                 2              0           1521           1323
Markdown                         8            101              0            362
YAML                             3              8             20            212
Bourne Shell                     6             25             30             74
INI                              1              0              0              3
-------------------------------------------------------------------------------
SUM:                           119           2255           3083          15933
-------------------------------------------------------------------------------
```

Not fun! Yet several teams have this experience over and over again most notably the Crypten team had to rewrite autograd from scratch.

Getting your own Dispatch key is even harder since every op you call goes through the dispatcher at leastt once. Understandably, not everyone is happy with adding new dispatch keys because it's a guaranteed x% slowdown on everyone else.

However, we do have something better we're excited about introducing `__torch__dispatch`


## The new world

So core has implementation of a `BaseTensor` which we can inherit from to create our own tensors.

We maintain a long list of custom `Tensor` implementations here https://github.com/albanD/subclass_zoo but we can dial in on a few good educational examples.

```python
# TODO: Can we do something simpler like a diagonal tensor implementation, quaternion, just do stuff that would for example resonate with Julia crowd. The existing examples feel more relevant to engineers than numerical folks.
```

## Sparse Tensors

Let's start with a simple example https://github.com/albanD/subclass_zoo/blob/main/sparse_output.py

Suppose we'd like to multiply two diagonal matrices and make the output sparse

```python
x = torch.randn(3, requires_grad=True)
y = torch.randn(3, requires_grad=True)
x, y = torch.diag(x), torch.diag(y)
```

We'd like to implement a function 

```python
r = sparse_output(torch.mul, x, y)
```

And as a baseline `sparse_output()` should be roughly equivalent to the core `torch.sparse_coo_tensor()`

And now here's when we use `__torch__dispatch__` 

```python
class SparseOutputMode(torch.Tensor):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func == torch.ops.aten.mul:

            # TODO: What's going on here
            r = super().__torch_dispatch__(func, types, args, kwargs)
            # TODO: Isn't `to_sparse` a core feature and defeats the purpose
            return r.to_sparse()
```



## Negative Tensor
As a baseline in April of 2021 we added a new Dispatch key for Negative Tensors https://github.com/pytorch/pytorch/pull/56058. This implementation makes it so we don't need instantiate a new Tensor with negative values but interpret its view as a negative Tensor. (TODO: Does this make any sense?)

This feature required changing 39 files in PyTorch core. Instead we can now implement it using `__torch__dispatch__`

We're interested in writing a new Tensor type `NegativeTensor` such that `NegativeTensor(torch.tensor(1)) == NegativeTensor(torch.tensor(-1))`

```python
class NegativeTensor(Tensor):
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(t):
            if isinstance(t, cls):
                with no_dispatch():
                    return t.neg() #TODO: I'm also confused here this seems to be implemented in core with its own dispatch key?
            else:
                return t

        return func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))


```


## Other Tensors
* Wrapper Tensor
* Meta Tensor
* More numerical tensors? E.g: quaternion, diagonal etc..
