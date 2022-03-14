# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# This file compares PyTorch's implementation of double backwards (which shares
# the same tape through multiple levels of differentiation) with a JAX-style
# nested grad implementation (which maintains a separate tape per level).
# The goals of this file are:
#
# 1. To make you aware of this difference in tape representation
# 2. To explain under what situations PyTorch's optimization is valid (grad-grad
#    versus grad-vmap-grad)
# 3. To describe a simple composition mechanism for eager mode JAX-style
#    transformations, eschewing the traditional "wrapper tensors" by having
#    a single hierarchy of dispatcher objects which forward each other.
#
# The autograd implementation and examples are directly cribbed from Zachary
# DeVito's Simple Autograd at
# https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC?usp=sharing
# Like in Zach's colab, we do not use PyTorch's autograd system at all.
#
# To get started, we replicate some of the data structures and helper functions
# from Simple Autograd.

# +
import torch
from torch import Tensor
from typing import List, NamedTuple, Callable, Dict, Optional


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs: List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: Callable[List[Tensor], List[Tensor]]


_name = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


def _dim_set(max_dim, d):
    if d is None:
        return tuple(range(0, max_dim))
    elif isinstance(d, int):
        return (d,)
    else:
        return tuple(sorted(d))


# -

# Unlike Simple Autograd, we won't make use of a Variable wrapper class.  For
# debuggability purposes, however, we still need a way to identify variables by
# a human readable name, and we'll do this by directly setting a t_name
# attribute on them.  variable will be called whenever we directly allocate
# a new tensor from PyTorch.


def variable(t: Tensor, name: str = None):
    if not hasattr(t, "t_name"):
        t.t_name = name or fresh_name()
    return t


# Instead of having a wrapper around each tensor individually, we will instead
# have a separate object, a Dispatcher object, which actually implements the
# supported operations on a tensor as a method that explicitly takes all the
# tensors as arguments.  If you are familiar with the history of ATen, the
# original implementation of ATen, these correspond to the
# CPUType/CUDAType/VariableType objects from that implementation (this was
# replaced with the modern dispatcher as the original vtable-based
# implementation did not support adding custom operators.)
#
# To start with, we will implement a backend dispatch layer Torch.  This just
# forwards on the operator calls to our underlying library PyTorch (and ensures
# that all the allocated tensors are labeled).  You could also imagine replacing
# this with a Numpy backend or even a pure Python variant (although this file is
# not currently setup to do so.)

# This is a little helper class that just defines dim in terms of size for
# convenience.  All of our dispatchers will inherit from it.
class Dispatcher:
    def dim(self, input):
        return len(self.size(input))

    # We could insert abstract methods for all the operations we expect here


class Torch(Dispatcher):
    # Here are the original four operators which were in Simple Autograd
    def mul(self, lhs, rhs):
        return variable(torch.mul(lhs, rhs))

    def add(self, lhs, rhs):
        return variable(torch.add(lhs, rhs))

    # Sum has been generalized to take an optional dim argument, which we
    # will need for Batched tensors
    def sum(self, input, dim=None, name=None):
        if dim is None:
            return variable(torch.sum(input), name)
        else:
            return variable(torch.sum(input, dim), name)

    def expand(self, input, sizes):
        return variable(input.expand(sizes))

    # For closure under Batched tensors, we need these operations...
    def unsqueeze(self, input, dim):
        return variable(torch.unsqueeze(input, dim))

    def squeeze(self, input, dim):
        return variable(torch.squeeze(input, dim))

    # ...and we also need to overload the meaning of size/ones to
    # hide/reinsert batch dimensions
    def size(self, input):
        return tuple(input.size())

    def ones(self, size):
        return variable(torch.ones(size))


# For debugging, it is helpful to print out what operations are occurring.  This
# decorator does just that!
class Logger(Dispatcher):
    def __init__(self, inner, *, name):
        self.inner = inner
        self.name = f"  {name}"

    def size(self, input):
        # don't log size calls
        return self.inner.size(input)

    def ones(self, size):
        r = self.inner.ones(size)
        print(f"{self.name} {r.t_name}: {self.size(r)} = ones({size})")
        return r

    def mul(self, lhs, rhs):
        r = self.inner.mul(lhs, rhs)
        if isinstance(rhs, float):
            print(f"{self.name} {r.t_name}: {self.size(r)} = {lhs.t_name} * {rhs}")
        else:
            print(
                f"{self.name} {r.t_name}: {self.size(r)} = {lhs.t_name} * {rhs.t_name}"
            )
        return r

    def add(self, lhs, rhs):
        r = self.inner.add(lhs, rhs)
        print(f"{self.name} {r.t_name}: {self.size(r)} = {lhs.t_name} + {rhs.t_name}")
        return r

    def sum(self, input, dim=None, name=None):
        r = self.inner.sum(input, dim=dim, name=name)
        print(f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.sum(dim={dim})")
        return r

    def unsqueeze(self, input, dim):
        r = self.inner.unsqueeze(input, dim)
        print(
            f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.unsqueeze({dim})"
        )
        return r

    def squeeze(self, input, dim):
        r = self.inner.squeeze(input, dim)
        print(f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.squeeze({dim})")
        return r

    def expand(self, input, sizes):
        r = self.inner.expand(input, sizes)
        print(
            f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.expand({sizes})"
        )
        return r


# The Autograd dispatcher is a wrapper around another dispatcher.  In the
# most typical situation, Autograd wraps Torch, taking the basic (non-autograd
# aware) tensor implementation and adds autograd support to it (delegating
# primitive operations to the inner dispatcher.  However, it doesn't have to
# wrap Torch!


class Autograd(Dispatcher):
    # create_graph here corresponds to the create_graph kwarg in traditional
    # PyTorch, which controls whether or not the graph of the derivative
    # will be constructed, allowing computing higher order derivatives.
    # We will see that although create_graph=True allows Autograd to directly
    # support higher order derivatives, layering an Autograd to another
    # Autograd will also allow higher order derivatives.
    def __init__(self, inner, *, name="Autograd", create_graph: bool = False):
        self.inner = inner
        self.gradient_tape = []
        self.name = name
        self.create_graph = create_graph

    # create_graph controls where add/mul/etc calls from the backwards
    # propagators go: if you create_graph, they recursively call back to
    # the current Autograd dispatcher; otherwise they move on to the inner
    # layer.
    def backward_inner(self):
        if self.create_graph:
            return self
        else:
            return self.inner

    def mul(self, lhs, rhs):
        if isinstance(rhs, float) and rhs == 1.0:
            # peephole optimization
            return lhs

        # define forward
        # first, run the operation in the inner layer to get the initial
        # result
        r = self.inner.mul(lhs, rhs)
        # We directly implement printing here as it indicates whether or not
        # this operation was saved to the tape or not
        print(f"{self.name} {r.t_name}: {self.size(r)} = {lhs.t_name} * {rhs.t_name}")

        # record what the inputs and outputs of the op were
        inputs = [lhs.t_name, rhs.t_name]
        outputs = [r.t_name]

        # define backprop
        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_doutputs

            dr_dlhs = rhs  # partial derivative of r = lhs*rhs
            dr_drhs = lhs  # partial derivative of r = lhs*rhs

            # chain rule propagation from outputs to inputs of multiply.
            # Notice that the propagation rule may itself call
            # other operations; depending on create_graph, they may
            # either go to self or self.inner; self.backward_inner()
            # controls which one we go to.
            dL_dlhs = self.backward_inner().mul(dL_dr, dr_dlhs)
            dL_drhs = self.backward_inner().mul(dL_dr, dr_drhs)
            dL_dinputs = [dL_dlhs, dL_drhs]
            return dL_dinputs

        # finally, we record the compute we did on the tape
        self.gradient_tape.append(
            TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate)
        )
        return r

    # The rest of the implementations follow in the same way and can
    # be skipped

    def add(self, lhs, rhs):
        # Add follows a similar pattern to Mul, but it doesn't end up
        # capturing any variables.
        r = self.inner.add(lhs, rhs)
        print(f"{self.name} {r.t_name}: {self.size(r)} = {lhs.t_name} + {rhs.t_name}")

        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_doutputs
            dr_dlhs = 1.0
            dr_drhs = 1.0
            dL_dlhs = self.backward_inner().mul(dL_dr, dr_dlhs)
            dL_drhs = self.backward_inner().mul(dL_dr, dr_drhs)
            return [dL_dlhs, dL_drhs]

        self.gradient_tape.append(
            TapeEntry(
                inputs=[lhs.t_name, rhs.t_name], outputs=[r.t_name], propagate=propagate
            )
        )
        return r

    # Extended to handle dim argument for Batched (later)
    def sum(self, input: Tensor, dim=None, name: Optional[str] = None):
        r = self.inner.sum(input, dim=dim, name=name)
        print(f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.sum(dim={dim})")

        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_doutputs
            size = self.inner.size(input)
            res = dL_dr
            # Broadcast over all dimensions that were reduced over
            for i in _dim_set(self.inner.dim(input), dim):
                res = self.backward_inner().unsqueeze(res, i)
            return [self.backward_inner().expand(res, size)]

        self.gradient_tape.append(
            TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate)
        )
        return r

    # Unlike Simple Autograd, this expand requires the input to have
    # been unsqueezed before hand.  This lets us avoid having to do
    # at::sum_to for the nontrivial case (which is more complicated)
    def expand(self, input: Tensor, sizes: List[int]):
        assert self.inner.dim(input) == len(sizes)  # only works if dims match
        r = self.inner.expand(input, sizes)
        print(
            f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.expand({sizes})"
        )

        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_doutputs
            input_size = self.inner.size(input)
            dims = tuple(
                i for i in range(self.inner.dim(input)) if input_size[i] != sizes[i]
            )
            # We wanted a sum keepdim=True, but I didn't want to force
            # everyone to support it so manually unsqueeze
            res = self.backward_inner().sum(dL_dr, dims)
            for d in dims:
                res = self.backward_inner().unsqueeze(res, d)
            return [res]

        self.gradient_tape.append(
            TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate)
        )
        return r

    # Unsqueeze are required for sum backwards, and then squeeze is required
    # for closure.  Size needed for batched tensor to modify size.

    def size(self, input: Tensor):
        return self.inner.size(input)

    def squeeze(self, input: Tensor, dim):
        r = self.inner.squeeze(input, dim)
        print(
            f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.squeeze(dim={dim})"
        )

        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_outputs
            return [self.backward_inner().unsqueeze(dL_dr, dim)]

        self.gradient_tape.append(
            TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate)
        )
        return r

    def unsqueeze(self, input: Tensor, dim):
        r = self.inner.unsqueeze(input, dim)
        print(
            f"{self.name} {r.t_name}: {self.size(r)} = {input.t_name}.unsqueeze(dim={dim})"
        )

        def propagate(dL_doutputs: List[Tensor]):
            (dL_dr,) = dL_doutputs
            return [self.backward_inner().squeeze(dL_dr, dim)]

        self.gradient_tape.append(
            TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate)
        )
        return r

    def ones(self, size):
        return self.inner.ones(size)

    def grad(self, L, desired_results: List[Tensor]) -> List[Tensor]:
        # this map holds dL/dX for all values X
        dL_d: Dict[str, Tensor] = {}
        # It starts by initializing the 'seed' dL/dL, which is 1
        # TODO: indirect this via the backend
        dL_d[L.t_name] = self.inner.ones(())
        print(f"-- {self.name} d{L.t_name} -------")

        # look up dL_dentries. If a variable is never used to compute the loss,
        # we consider its gradient None, see the note below about zeros for more information.
        def gather_grad(entries: List[str]):
            return [dL_d[entry] if entry in dL_d else None for entry in entries]

        # propagate the gradient information backward
        for entry in reversed(self.gradient_tape):
            dL_doutputs = gather_grad(entry.outputs)
            if all(dL_doutput is None for dL_doutput in dL_doutputs):
                # optimize for the case where some gradient pathways are zero. See
                # The note below for more details.
                continue

            # perform chain rule propagation specific to each compute
            dL_dinputs = entry.propagate(dL_doutputs)

            # Accululate the gradient produced for each input.
            # Each use of a variable produces some gradient dL_dinput for that
            # use. The multivariate chain rule tells us it is safe to sum
            # all the contributions together.
            for input, dL_dinput in zip(entry.inputs, dL_dinputs):
                if input not in dL_d:
                    dL_d[input] = dL_dinput
                else:
                    dL_d[input] = self.backward_inner().add(dL_d[input], dL_dinput)

        # print some information to understand the values of each intermediate
        # for name, value in dL_d.items():
        #     print(f'{self.name} d{L.t_name}_d{name} = {value.t_name}')
        print(f"------------------------")

        return gather_grad(desired.t_name for desired in desired_results)


# To calculate some simple gradients, we can compose Autograd with
# Torch and get the result we expect.

# +
torch.manual_seed(0)
a, b = variable(torch.rand(4)), variable(torch.rand(4))


def simple(d, a, b):
    t = d.add(a, b)
    return d.mul(t, b)


d = Autograd(Torch())

loss = simple(d, a, b)
da, db = d.grad(loss, [a, b])
print("da", da)
print("db", db)
# -

# To compute higher order gradients, we have two options.  First,
# we can do traditional PyTorch style higher order differentiation
# with `create_graph=True`, writing the backpropagation computations directly
# into the tape so they can be further differentiated over.  This is also
# what the original Simple Autograd implementation does.

# +
d = Autograd(Torch(), create_graph=True)

# I slightly generalized this function so that it works for the next
# example; d2 is the dispatcher run on the first grad call, and d1 is
# for the second (we'll see why the numbers are inverted shortly).
def run_gradients(d2, d1):
    # our first loss
    L0 = d2.sum(simple(d2, a, b), name="L0")

    # compute derivatives of our inputs
    dL0_da, dL0_db = d2.grad(L0, [a, b])

    # In real code, how would we switch from executing from d2 to d1?
    # In functorch, the d2 dispatch calls would happen in the inside of
    # a higher-order grad() call; when we exit from this call, all
    # of the involved tensors are unwrapped.

    # now lets compute the L2 norm of our derivatives
    L1 = d1.sum(d1.add(d1.mul(dL0_da, dL0_da), d1.mul(dL0_db, dL0_db)), name="L1")

    # and take the gradient of that.
    # notice there are two losses involved1.
    dL1_da, dL1_db = d1.grad(L1, [a, b])
    return dL1_da, dL1_db


da, db = run_gradients(d, d)
print("da", da)
print("db", db)
# -

# Our second option is to follow functorch's implementation strategy, which
# is to stack two Autograd dispatchers on top of each other.  Here, it is
# not necessary to `create_graph=True`, because when the backpropagator forwards
# to the inner dispatcher, it will record those operations on the tape too.
# But if you look at the output, you will notice something very interesting:
# the first portion of the tape is exactly replicated between Autograd1 and
# Autograd2: we're duplicating the tape in this case!  So PyTorch's default
# implementation of backwards is more efficient, because it avoids having to
# record the tape twice (although this doesn't matter too much, because the
# saved tensors themselves can be shared between the two tapes, so it is just
# the operator graph that is duplicated.

# +
# turning off create_graph will impede us from seeing the logging lines for
# the second backwards, so we turn on logging for Torch to see them
d1 = Autograd(Logger(Torch(), name="Torch"), name="Autograd1", create_graph=False)
d2 = Autograd(d1, name="Autograd2", create_graph=False)

da, db = run_gradients(d2, d1)
print("da", da)
print("db", db)
# -

# Under what situations might it be profitable to keep the two tapes separate?
# One guess we might have is if there is another functional transformation
# wedged between the two autograd transformations.  We would then expect the
# backwards formula we save to be different between the two tapes.  To do this, I
# first need to implement batched tensors.
#
# One unusual thing about this implementation is that we do not need to wrap
# tensors to change their sizes; instead, we just override the meaning of
# size() on the dispatcher to hide batch dimensions.  One case we do not
# exercise in this example is implicit broadcasting when you combine a tensor
# that is not batched with a tensor that is batched: without wrappers, a user
# must explicitly lift (e.g., unsqueeze and expand) tensors they wish to
# replicate across the batch dimension.  The code below will blindly attempt to
# reinterpret a tensor as a batched tensor, even when it may not make sense (if
# there is a size mismatch, however, you will get an assert failure).  Similarly,
# once you exit a vmap region, all previously vmap'ed tensors "magically" become
# unbatched.  functorch did not pursue this implementation because at the time
# Tensor.size() was not virtual and thus it was not possible to override (this
# will be changing soon).

# This implementation of Batched only supports inserting a dimension
# at the very front
class Batched(Dispatcher):
    def __init__(self, inner, *, length, name="Batched"):
        self.inner = inner
        self.name = name
        self.length = length

    def size(self, input):
        sizes = self.inner.size(input)
        assert sizes[0] == self.length
        return sizes[1:]

    def ones(self, size):
        return self.inner.ones((self.length,) + size)

    def mul(self, lhs, rhs):
        assert self.inner.size(lhs)[0] == self.length
        if not isinstance(rhs, float):
            assert self.inner.size(rhs)[0] == self.length
        return self.inner.mul(lhs, rhs)

    def add(self, lhs, rhs):
        assert self.inner.size(lhs)[0] == self.length
        assert self.inner.size(rhs)[0] == self.length
        return self.inner.add(lhs, rhs)

    def sum(self, input, dim=None, name=None):
        # offset all the summed over dimensions by one
        assert self.inner.size(input)[0] == self.length
        dim = tuple(i + 1 for i in _dim_set(self.inner.dim(input) - 1, dim))
        return self.inner.sum(input, dim, name=name)

    def expand(self, input, sizes):
        # offset sizes by one
        assert self.inner.size(input)[0] == self.length
        return self.inner.expand(input, (self.inner.size(input)[0],) + sizes)

    def squeeze(self, input, dim):
        # offset dim by one
        assert self.inner.size(input)[0] == self.length
        return self.inner.squeeze(input, dim + 1)

    def unsqueeze(self, input, dim):
        # offset dim by one
        assert self.inner.size(input)[0] == self.length
        return self.inner.unsqueeze(input, dim + 1)


# +
# Our inputs are batched this time!
va, vb = variable(torch.rand(2, 4)), variable(torch.rand(2, 4))

d1 = Autograd(Torch(), name="Autograd1", create_graph=False)
d2 = Batched(d1, length=2, name="Batched2")
d3 = Autograd(d2, name="Autograd3", create_graph=False)


def run_batched_gradients(d3, d2, d1):
    # our first loss
    # we write the dimension we reduce on explicitly for clarity
    L0 = d3.sum(simple(d3, va, vb), dim=0, name="L0")

    # compute derivatives of our inputs
    dL0_da, dL0_db = d3.grad(L0, [va, vb])

    # now lets compute the L2 norm of our derivatives
    L1 = d1.sum(d1.add(d1.mul(dL0_da, dL0_da), d1.mul(dL0_db, dL0_db)), name="L1")

    # and take the gradient of that.
    # notice there are two losses involved1.
    dL1_da, dL1_db = d1.grad(L1, [va, vb])
    return dL1_da, dL1_db


dva, dvb = run_batched_gradients(d3, d2, d1)
print("va", va)
print("vb", vb)
print("dva", dva)
print("dvb", dvb)
# -

# To see that we have done this correctly, we could run the corresponding JAX:
#
# ```
# from jax import grad, vmap
# import jax.numpy as np
#
# def simple(a, b):
#   t = a + b
#   return t * b
#
# def L0(a, b):
#   return np.sum(simple(a, b))
#
# def L1(a, b):
#   dL0_da, dL0_db = vmap(grad(L0, argnums=(0,1)), in_axes=0)(a, b)
#   return (dL0_da * dL0_da + dL0_db * dL0_db).sum()
#
# va = np.asarray([[0.4556, 0.6323, 0.3489, 0.4017],
#         [0.0223, 0.1689, 0.2939, 0.5185]])
# vb = np.asarray([[0.6977, 0.8000, 0.1610, 0.2823],
#         [0.6816, 0.9152, 0.3971, 0.8742]])
# dva, dvb = grad(L1, argnums=(0,1))(va, vb)
# print("dva", dva)
# print("dvb", dvb)
# ```
#
# Looking over the output, the tapes look similar, but we can see that the sizes
# and the arguments of the operations in question differ (after all, Autograd3 is
# on the inside of the vmap, while Autograd1 is outside).  But it is still very
# similar: we could imagine simply varying the dispatcher we use to process backwards
# depending on when we are executing the tape.  In fact, this is exactly what an
# initial, non-functorch implementation of PyTorch did to support per-sample
# gradients.
#
# Exercise: modify Autograd.grad to accept a dispatcher, and use that dispatcher
# instead of self.backward_inner() when running propagator functions.  Then, rewrite
# the above example so that it only has one level of Autograd:
# Batched(Autograd(Torch(), create_graph=True)) and show you still get the same
# result.
#
# Epilogue: Why didn't I use inheritance?  In fact, in the first version
# of this notebook, I did.  But self makes it easy to accidentally go
# back to the "top" of the dispatch stack, which is typically not what
# you want.