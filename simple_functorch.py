# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# This notebook walks through a self-contained implementation of
# functorch, including support for both vjp and vmap combinators (using
# PyTorch only to implement primitive tensor operations).  It follows
# the tradition of
# [Autodidax](https://jax.readthedocs.io/en/latest/autodidax.html) (a
# pedagogical reimplementation of JAX, the library functorch is inspired
# by) and [Simple
# Autograd](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC?usp=sharing)
# (Zachary Devito's pedagogical reimplementation of autograd, which the
# autograd system in this notebook is based off of.) You can [open this
# file in
# Colab](https://colab.research.google.com/github/albanD/subclass_zoo/blob/main/simple_functorch.ipynb)
# and play around with the examples.
#
# As a simplified implementation of functorch, this notebook also makes
# it easier to investigate some more subtle aspects of how PyTorch's
# native autograd system interacts with composable transforms.  In
# particular, we will see that PyTorch's native implementation of double
# backwards (which shares the same tape through multiple levels of
# differentiation) differs from functorch's nested grad implementation
# (which maintains a separate tape per level).

# To get started, we replicate some of the data structures and helper functions
# from Simple Autograd.

# +
import contextlib
import functools
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional

import torch
from torch import Tensor
from torch.utils._pytree import tree_map


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs: List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: Callable[[List[Tensor]], List[Tensor]]


_name = 0


def fresh_name() -> str:
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


# -

# This is a little helper function for converting the dim argument in
# sum into an explicit list of dimensions that will be reduced over.
# It takes the dim of the tensor we are summing over and the dim
# argument itself.


def sum_dims(*, input_dim, dim):
    if dim is None:
        return tuple(range(0, input_dim))
    elif isinstance(dim, int):
        return (dim,)
    else:
        return tuple(sorted(dim))


# In Simple Autograd, we provided a Variable wrapper class which
# provided a traditional Tensor style interface for our objects; in
# functorch proper, objects are repeatedly wrapped in this way to
# implement multipler layers of transformations.
#
# In my opinion, this sort of wrapper makes it more difficult to
# understand the flow of logic.  So in Simple Functorch, we take a
# different approach: we won't make use of a wrapper class at all,
# instead showing how to add it in the end as syntax sugar on top of our
# system.
#
# For debuggability purposes, however, it is nice to have a way to
# identify variables by a human readable name.  We'll do this by setting
# a t_name attribute on PyTorch tensors whenever we allocate a new
# tensor.


def label(t: Tensor, name: str = None):
    if not hasattr(t, "t_name"):
        t.t_name = name or fresh_name()
    return t


# So if we aren't going to have a wrapper around each tensor, how will
# we actually implement our logic?  We will organize our various layers
# of transformations as separate Dispatcher objects, which define
# methods for performing operations on tensors, but are not Tensors
# themselves.  For example, instead of defining Tensor.add(Tensor), we
# will define Dispatcher.add(Tensor, Tensor).  If you are familiar with
# historical ATen, in the original implementation of ATen, these
# correspond to the CPUType/CUDAType/VariableType objects from that
# implementation (this was replaced with the modern dispatcher as the
# original vtable-based implementation did not support adding custom
# operators.)


class Dispatcher:
    inner = None

    def mul(self, lhs, rhs):
        raise NotImplementedError

    def add(self, lhs, rhs):
        raise NotImplementedError

    # Sum has been generalized to take an optional dim argument, which we
    # will need for Batched tensors
    def sum(self, input, dim=None, name=None):
        raise NotImplementedError

    def expand(self, input, sizes):
        raise NotImplementedError

    # For closure under Batched tensors, we need these operations...
    def unsqueeze(self, input, dim):
        raise NotImplementedError

    def squeeze(self, input, dim):
        raise NotImplementedError

    # ...and we also need to overload the meaning of size/ones to
    # hide/reinsert batch dimensions.  We also introduce a concept
    # of "lifting" a tensor to be batched by broadcasting it on
    # a dimension
    def size(self, input):
        raise NotImplementedError

    def ones(self, size):
        raise NotImplementedError

    def lift(self, input, d):
        raise NotImplementedError

    # For convenience, we provide dim, which just returns the length of
    # the sizes
    def dim(self, input):
        return len(self.size(input))


# To start with, we can implement a backend dispatcher layer Torch,
# which just forwards the operator calls to our underlying library PyTorch
# (and ensures that all the allocated tensors are labeled with variable).  You could
# also imagine replacing this with a Numpy backend or even a pure Python
# variant (although this file is not currently setup to do so.)


class Torch(Dispatcher):
    def mul(self, lhs, rhs):
        return label(torch.mul(lhs, rhs))

    def add(self, lhs, rhs):
        return label(torch.add(lhs, rhs))

    def sum(self, input, dim=None, name=None):
        if dim is None:
            return label(torch.sum(input), name)
        else:
            return label(torch.sum(input, dim), name)

    def expand(self, input, sizes):
        return label(input.expand(sizes))

    def unsqueeze(self, input, dim):
        return label(torch.unsqueeze(input, dim))

    def squeeze(self, input, dim):
        return label(torch.squeeze(input, dim))

    def size(self, input):
        # Return size a tuple for marginally more compact printing
        assert isinstance(input, torch.Tensor)
        return tuple(input.size())

    def ones(self, size):
        return label(torch.ones(size))

    def custom_vjp(self, fwd_fn, bwd_fn, *args):
        # The backend layer for custom_vjp just calls fwd_fn.
        # Why doesn't it create an autograd.Function? We're assuming the backend
        # layer doesn't need to handle Autograd.
        a, b = fwd_fn(self, *args)
        result = label(a), label(b)
        return result

    def custom_vmap(self, fn, batch_rule, *args):
        results = fn(self, *args)
        return results

    def lift(self, input, d):
        assert d is self
        return input


# Dispatcher layers are composable via object composition: we can
# imagine a stack of dispatchers, each one calling into the next.
# For example, the Logger dispatcher simply prints out what operation
# was called on it, and then forwards on the operation to the inner
# dispatcher.


def custom_vjp_str(r, fwd_fn, bwd_fn, args):
    arg_names = ", ".join([a.t_name for a in args])
    r_is_tensor = isinstance(r, torch.Tensor)
    if r_is_tensor:
        result_names = r.t_name
    else:
        result_names = [r.t_name for r in r]
        if len(result_names) == 1:
            result_names = f"{result_names[0]},"
        else:
            result_names = ", ".join(result_names)

    print(
        f"{result_names} = custom_vjp({fwd_fn.__name__}, {bwd_fn.__name__}, {arg_names})"
    )


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

    def custom_vjp(self, fwd_fn, bwd_fn, *args):
        r = self.inner.custom_vjp(fwd_fn, bwd_fn, *args)
        print(custom_vjp_str(r, fwd_fn, bwd_fn, args))
        return r

    def lift(self, input, d):
        if d is self:
            return input
        return self.inner.lift(input, d)


# Here is a simple example of using Logger and Torch together.  Whenever
# we make calls to operations, we must do so via the Dispatcher object.
# We will explicitly write out all of these calls before we add wrapper
# class sugaring.

d = Logger(Torch(), name="Torch")
print(d.add(d.ones(2), d.ones(2)))


# With the Dispatcher structure in hand, we are now in a good place to
# port the autograd implementation from Simple Autograd into our new
# framework.


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
            for i in sum_dims(input_dim=self.inner.dim(input), dim=dim):
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

    def custom_vjp(self, fwd_fn, bwd_fn, *args):
        # To support Autograd(Autograd(Torch()), custom_vjp MUST call custom_vjp
        # on the inner dispatcher. If it instead called fwd_fn(*args), then
        # the inner Autograd dispatcher would not use bwd_fn in its backward pass.
        r, saved = self.inner.custom_vjp(fwd_fn, bwd_fn, *args)
        print(custom_vjp_str(r, fwd_fn, bwd_fn, args))

        # To preserve custom backward semantics, we create a lambda that calls
        # bwd_fn. This lambda is then saved on the gradient tape.
        def propagate(dL_doutputs: List[Tensor]):
            return bwd_fn(self, dL_doutputs, saved)

        self.gradient_tape.append(
            TapeEntry(
                inputs=[arg.t_name for arg in args],
                outputs=[r.t_name],
                propagate=propagate,
            )
        )
        return r, saved

    def custom_vmap(self, fn, batch_rule, *args):
        def call_with_current_dispatcher(fn):
            def wrapped(d, *args):
                saved = self.inner
                try:
                    self.inner = d
                    result = fn(self, *args)
                    return result
                finally:
                    self.inner = saved
            return wrapped

        # either fn or batch_rule gets invoked later down the line. Whichever one
        # it is, we want to record the history onto this dispatcher's gradient tape.
        result = self.inner.custom_vmap(
            call_with_current_dispatcher(fn),
            call_with_current_dispatcher(batch_rule), *args)
        return result

    def lift(self, input, d):
        if d is self:
            return input
        return self.inner.lift(input, d)

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
a, b = label(torch.rand(4)), label(torch.rand(4))


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
#
# This is our first example of using two dispatchers.  While we are
# performing the inner grad, we perform our operations on the outer
# dispatcher `d2`; after we are done with the inner grad we switch to
# `d1`.  Intuitively, this corresponds from passing out of the inner
# `grad` call to the outer `grad` call.

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
        dim = tuple(
            i + 1 for i in sum_dims(input_dim=self.inner.dim(input) - 1, dim=dim)
        )
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

    def custom_vjp(self, fwd_fn, bwd_fn, *args):
        def batchify(fn):
            def new_fn(d, *args):
                new_d = Batched(d, length=self.length, name="GeneratedBatched")
                result = fn(new_d, *args)
                return result

            return new_fn

        # If we have Batched(Autograd(Torch()), then we would like the inner
        # dispatcher to receive a call to custom_vjp so that it preserves the
        # backward semantics. However, since this is the Batched dispatcher,
        # we want the innermost Torch dispatcher to run a batched version of fwd_fn
        # function! The way we get this to work is to create a new fwd_fn, that,
        # when executed, executes a batched version of fwd_fn.
        #
        # Same thing for the bwd_fn.
        # NB: currently simple_functorch assumes that all Tensors are batched at
        # dimension 0. I'm not sure how this logic would look like without
        # this assumption (in functorch tensors may not be batched).
        r, saved = self.inner.custom_vjp(batchify(fwd_fn), batchify(bwd_fn), *args)
        return r, saved

    def custom_vmap(self, fn, batch_rule, *args):
        result = batch_rule(self.inner, *args)
        return result

    # The lift operation takes a tensor associated with some inner
    # dispatcher, and "lifts" it so that it is interpreted neutrally
    # for the outer dispatcher.  For most dispatchers this is trivial,
    # but for batched tensor it is not: given a tensor x, to interpret
    # it as x under the Batching dispatcher, we have to expand it so
    # that it is broadcasted along its first dimension.
    def lift(self, input, d):
        if d is self:
            return input
        b_input = self.inner.unsqueeze(input, 0)
        b_input = self.inner.expand(b_input, (self.length,) + self.inner.size(input))
        return self.inner.lift(b_input, d)


# +
# Our inputs are batched this time!
va, vb = label(torch.rand(2, 4)), label(torch.rand(2, 4))

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


# OK, so all of this dispatcher business is all nice and explicit, but
# that's not what JAX/functorch's interface looks like.  How do we
# bridge the gap?  Our first problem is heving to explicitly thread
# our Dispatcher object everywhere.  In functorch, we instead implicitly
# have a "current mode" which changes when you enter a grad() or vmap()
# function.  So let's maintain global current dispatcher, and a way to
# change what the current dispatcher is.  You can think of this as a
# singly-linked stack of dispatchers which we push and pop dispatchers
# onto.

# +
DISPATCHER = Torch()


@contextlib.contextmanager
def dispatcher(d):
    global DISPATCHER
    old_d = DISPATCHER
    DISPATCHER = d
    try:
        yield
    finally:
        DISPATCHER = old_d


# -

# A dispatcher mode, however, is not enough.  Remember that in our
# implementation of Batched, we blindly assumed that all tensors were
# batched, even if this did not necessarily make sense.  If I have
# `vmap(lambda bx: bx + y)(x)`, with `x: (B,X)` and `y: (X,)`, the
# underlying operation should broadcast y to `(B,X)` and then do the
# addition with x (bx advertises that it has size `(X,)` inside of the
# vmap'd lambda).  To know this should happen, it is necessary for
# us to know that y is not a batched tensor, but x is a batched tensor.
# We'll resolve this with a wrapper class called FuncTensor, which
# records both the underlying Tensor, as well as the Dispatcher which
# this tensor is associated with.  In the above example, `bx.dispatcher`
# might be `Batched(Torch())`, whereas `x.dispatcher` is `Torch()`.
#
# So our general strategy is as follows:
#   1. Every tensor is associated with a dispatcher
#   2. You can lift tensors to dispatchers which wrap them (which can
#      trigger some operations, like expand for Batched); this is
#      implemented by `dispatcher_wraps`
#   3. To perform an operation between to tensors, lift them so that
#      they all have the same dispatcher, then do the operation on
#      that dispatcher.

# +
# A dispatcher d1 wraps another dispatcher d2 if d2 is an ancestor of
# d1 in the tree structure.  We've defined this relation to be
# reflexive, in the same way issubclass(A, A) == True.
def dispatcher_wraps(d1, d2):
    # Treat this as a reflexive relation
    if d1 is d2:
        return True
    while d1.inner is not None:
        d1 = d1.inner
        if d1 is d2:
            return True
    return False


# Given a list of arguments, lift them all up to a common dispatcher
# level, returning that dispatcher as well as the lifted arguments.
# Note that the global current DISPATCHER is also accounted for!
# In autodidax, this is `find_top_trace`.
def lift_and_unwrap_args(*args):
    outermost = DISPATCHER
    for a in args:
        if dispatcher_wraps(outermost, a.dispatcher):
            pass
        elif dispatcher_wraps(a.dispatcher, outermost):
            # You can make this case an error as well if you don't
            # want to support non-lexical functorch tensors
            outermost = a.dispatcher
        else:
            raise TypeError("incompatible dispatcher trees")
    return (outermost,) + tuple(a.lift(outermost).tensor for a in args)


# -


# The actual implementation of the wrapper tensor which tracks the
# Dispatcher for a tensor


@dataclass
class FuncTensor:
    tensor: Tensor
    dispatcher: Dispatcher

    # Lift a FuncTensor to an outer dispatcher
    def lift(self, d):
        # You can only lift to a dispatcher which wraps the dispatcher
        # this FuncTensor is associated with (not vice versa, or between
        # unrelated FuncTensors).
        assert dispatcher_wraps(d, self.dispatcher)
        return FuncTensor(d.lift(self.tensor, self.dispatcher), d)

    # The general strategy for any operation performed on a tensor, we
    # lift all the arguments so that they live on the same dispatcher
    # level, and then perform the operation on that dispatcher.  The
    # resulting tensor is tagged at whatever dispatcher we had run the
    # tensor on.
    def __mul__(self, other):
        d, self, other = lift_and_unwrap_args(self, other)
        return FuncTensor(d.mul(self, other), d)

    def __add__(self, other):
        d, self, other = lift_and_unwrap_args(self, other)
        return FuncTensor(d.add(self, other), d)

    def sum(self, dim=None, name=None):
        d, self = lift_and_unwrap_args(self)
        return FuncTensor(d.sum(self, dim, name=name), d)

    def expand(self, sizes):
        d, self = lift_and_unwrap_args(self)
        return FuncTensor(d.expand(self, sizes), d)

    def unsqueeze(self, dim):
        d, self = lift_and_unwrap_args(self)
        return FuncTensor(d.unsqueeze(self, dim), d)

    def squeeze(self, dim):
        d, self = lift_and_unwrap_args(self)
        return FuncTensor(d.squeeze(self, dim), d)

    def size(self):
        d, self = lift_and_unwrap_args(self)
        return d.size(self)

    def dim(self):
        d, self = lift_and_unwrap_args(self)
        return d.size(self)

    # Factory functions like ones do not have any Tensor arguments,
    # so they rely solely on the ambient DISPATCHER to determine
    # what their semantics should be
    @staticmethod
    def ones(size):
        d = lift_and_unwrap_args()
        return d.ones(size)


# Now we are ready to implement grad.  First, we need some helper
# functions.

# +
# When we are done doing a vmap/grad, we need to take the results and
# lower them back to a lower dispatcher on the stack (this is always
# a no-op, in particular, in the vmap case, when we exit vmap the user
# gets to see the batched dimension again.)
def unlift(t, d):
    if isinstance(t, list):
        return [unlift(x, d) for x in t]
    elif isinstance(t, tuple):
        return tuple(unlift(x, d) for x in t)
    else:
        if t.dispatcher is d:
            return t
        return unlift(FuncTensor(t.tensor, t.dispatcher.inner), d)


# This lets us easily pick out arguments as specified by argnums
def filter_argnums(args, argnums):
    if isinstance(argnums, int):
        return (args[argnums],)
    else:
        return tuple(args[i] for i in argnums)


# -

# Now grad and vmap!

# For simplicity, these functions only take tuples, not pytrees
def grad(f, argnums=0):
    @functools.wraps(f)
    def wrapped_f(*args):
        # We first lift and unwrap all of the arguments which we want
        # to pass into the function
        old_d, *args = lift_and_unwrap_args(*args)
        d = Autograd(old_d)
        with dispatcher(d):
            # We pass in the functions at the new Autograd level (they
            # were lifted to old_d, and lifting to d is a noop)
            L = f(*(FuncTensor(a, d) for a in args))
            assert L.dispatcher is d
            # Run the autograd pass, getting the grads for the inputs
            # as specified by argnums
            grads = d.grad(L.tensor, filter_argnums(args, argnums))
            # Finally, construct the grads at the lower level and return
            # them
            return [FuncTensor(r, old_d) for r in grads]

    return wrapped_f


def vmap(f):
    @functools.wraps(f)
    def wrapped_f(*args):
        # cannot vmap over no arguments as this function uses the
        # arguments to determine how large the batch dimension is
        # (hypothetically, you could explicitly pass in the batch
        # size, and then use this to control factory functions;
        # JAX doesn't seem to have a knob to do this)
        assert args
        old_d, *args = lift_and_unwrap_args(*args)
        d = Batched(old_d, length=old_d.size(args[0])[0])
        for a in args:
            assert old_d.size(a)[0] == d.length
        with dispatcher(d):
            # Rewrap all the arguments as batched tensors, then
            # unwrap any batched tensors that escape
            return unlift(f(*(FuncTensor(a, d) for a in args)), old_d)

    return wrapped_f


# Small test: we want to make sure that we can run multiple layers of vmap and get
# the right behavior
x = FuncTensor(label(torch.randn(3, 4, 5)), DISPATCHER)
ret = vmap(vmap(lambda a: a.unsqueeze(0)))(x)
assert ret.size() == (3, 4, 1, 5)
assert torch.allclose(ret.tensor, x.unsqueeze(2).tensor)

# We should see a tensor of size (3, 4, 1, 5) because there's two layers of vmap
# it's the equivalent of unsqueeze(0) when the tensor is (5,) and then adding (3, 4) back on

# Now we can rerun our example using the high level grad/vmap functions!

# +
def simple(a, b):
    t = a + b
    return t * b


def L0(a, b):
    return simple(a, b).sum()


def L1(a, b):
    dL0_da, dL0_db = vmap(grad(L0, argnums=(0, 1)))(a, b)
    return (dL0_da * dL0_da + dL0_db * dL0_db).sum()


fva = FuncTensor(va, DISPATCHER)
fvb = FuncTensor(vb, DISPATCHER)
dva, dvb = grad(L1, argnums=(0, 1))(fva, fvb)
print("dva", dva)
print("dvb", dvb)
# -

# Because FuncTensors are associated with the ambient dispatcher they
# were created from, they are also allowed to escape from the context in
# which they were defined, allowing for non-lexical, imperative
# transform API.  For example, batching over module parameters is
# problematic today, but all we need to do is tweak the FuncTensor's
# dispatchers appropriately and everything works out.

# +

PlainTensor = lambda t: FuncTensor(torch.randn(N), DISPATCHER)
BatchedTensor = lambda t: FuncTensor(t, Batched(DISPATCHER, length=B))


class ScaleBiasModule:
    weight: FuncTensor
    bias: FuncTensor

    def __init__(self, N):
        self.weight = PlainTensor(torch.randn(N))
        self.bias = PlainTensor(torch.randn(N))

    def forward(self, input):
        return self.weight * input + self.bias


B = 2
N = 3
m = ScaleBiasModule(N)
# Ensemble weights only; input is not batched
m.weight = BatchedTensor(torch.randn(B, N))
input = PlainTensor(torch.randn(N))
output = m.forward(input)
print(
    "expect", input.tensor.unsqueeze(0) * m.weight.tensor + m.bias.tensor.unsqueeze(0)
)
print("output", output.tensor)

# Autodidax decoder ring
#
# Tracer = FuncTensor
#   can be subclassed for storing extra data (JVPTracer, BatchTracer)
# AbstractValue = this represents the "dtype" and the "sizes", etc (stored on Tracer)
#   It's a type! But it also has sizes (like a meta tensor)
#   ShapedArray/ConcreteArray (distinguish between device and meta
#   tensor) ~> going to bind
#   ...not really Dispatcher; represents the dtype/size stuff
# MainTrace = Dispatcher (as stored in DISPATCHER stack) (why do they
#   also need Trace? Weird.  as seen in jvp_v1 they first new_main
#   and then wrap it in JVPTrace)
# Trace = Dispatcher (the thing that gets subclassed to have
#   implementations, e.g., EvalTrace, JVPTrace)
#   pure = take a constant and make the Tracer for it
#   lift = take an inner Tracer and make it this level
#
# new_main ~> dispatcher, gives you the Dispatcher/MainTrace
# get_aval(t) ~> getting the dtype/size metadata stuff
# bind ~> inlined in FuncTensor methods, including lift_and_unwrap_args
# full_lower ~> not implemented
# full_raise ~> lift
# find_top_trace ~> lift_and_unwrap_args
#
# dynamic_trace ~> the thing that pushes jit to the bottom
#
# jitting gives you an xla_call_p, which STORES the jaxpr
#   transforms look into the jaxpr and retrace it with the transform

# -

# Higher order operators in simple_functorch!
#
# Problem: users want to define functions with custom forward and backward
# passes. These functions call PyTorch operations. When we vmap over such a
# function, we would like for the backward pass to be preserved.
#
# Why is this difficult? In PyTorch today, vmap over an autograd.Function
# effectively runs vmap on the forward pass of the autograd.Function.
# Meanwhile, autograd records the transformed operations for backward, instead
# of the custom backward pass we specified in the autograd.Function!
#
# Solution: We're going to introduce a `custom_vjp` primitive that accepts
# functions and varargs Tensor arguments and demonstrate that it resolves
# the problem.
#
# custom_vjp(fwd_fn, bwd_fn, *args) takes in two functions as arguments.

# +

# For our custom function, we want f(x) = x * x, but we install a custom
# backwards pass that computes 32 * x (instead of 2 * x) so we can tell
# if custom_vjp is working.

d = Autograd(Torch())

a = label(torch.rand(4))
va = label(torch.rand(2, 4))


def f_fwd(dispatcher, x):
    # Our convention is that f_fwd returns (outputs, "saved")
    return dispatcher.mul(x, x), x


# Our convention is that f_bwd accepts (dispatcher, gradOutputs, "saved")
def f_bwd(dispatcher, gradOutputs, x):
    (gO,) = gradOutputs
    # Should be gO * 2 * x, but we're gonna do gO * 32 * x to demonstrate things
    return [
        dispatcher.mul(dispatcher.mul(gO, x), label(torch.tensor(32.0), "thirty_two"))
    ]


# -

# Okay, now let's try it out!
#
# The implementaton of custom_vjp is:
# - call custom_vjp on the inner dispatcher
# - save f_bwd as a lambda onto the gradient tape
# (See earlier in this colab for more details)

# +

# A single layer of autograd
def run_grad(d):
    # Here's how to invoke custom_vjp.
    r, _ = d.custom_vjp(f_fwd, f_bwd, a)
    L3 = d.sum(r, name="L3")
    (dL3_a,) = d.grad(L3, [a])

    # Check that the gradients are indeed 32 * a
    assert torch.allclose(dL3_a, 32 * a)


run_grad(d)

# Multiple layers of autograd
def run_gradgrad(d2, d1):
    r, _ = d2.custom_vjp(f_fwd, f_bwd, a)
    L4 = d2.sum(r, name="L4")
    (dL4_a,) = d2.grad(L4, [a])

    # Evidence that d2 respected the custom_vjp's f_bwd
    assert torch.allclose(dL4_a, 32 * a)

    assert hasattr(dL4_a, "t_name")
    dL4_a_sum = d1.sum(dL4_a, name="dL4_a_sum")
    (ddL4_a_a,) = d1.grad(dL4_a_sum, [a])

    # Evidence that d1 respected the custom_vjp's f_bwd
    assert torch.allclose(ddL4_a_a, torch.ones_like(a) * 32)


d1 = Autograd(Logger(Torch(), name="Torch"), name="Autograd1", create_graph=False)
d2 = Autograd(d1, name="Autograd2", create_graph=False)
run_gradgrad(d2, d1)

# -

# And now, let's try that again, with grad(lambda x: vmap(f)(x).sum()).
# The goal of custom_vjp is to make it so that vmap(custom_vjp) still
# preserves the backward semantics.

# +


def f_fwd(d, x):
    return d.mul(x, x), x


def f_bwd(d, gradOutputs, x):
    (gO,) = gradOutputs
    # Should be gO * 2 * x, but we're gonna do gO * 32 * x to prove a point
    return [d.mul(d.mul(gO, x), label(torch.ones_like(x) * 32.0, "thirty_two"))]


d1 = Autograd(Torch(), name="Autograd1", create_graph=False)
d2 = Batched(d1, length=2, name="Batched2")


def run_gradvmap(d2: "Batched", d1: "Autograd"):
    r, _ = d2.custom_vjp(f_fwd, f_bwd, va)
    L99 = d1.sum(r, name="L99")
    (dL99_a,) = d1.grad(L99, [va])

    # As you can see, d1.grad still calls f_bwd.
    # The way we got this to work is that Batched.custom_vjp
    # calls custom_vjp on its inner dispatcher.
    # Scroll up to the implementation of Batched for more details.
    assert torch.allclose(dL99_a, 32 * va)


run_gradvmap(d2, d1)

d = Batched(Torch(), length=3)

# Custom vmap
def f(d, x):
    return d.mul(x, x)

def f_batch_rule(d, x):
    # to prove a point
    return d.add(x, x)

x = label(torch.tensor([1., 2., 3.]))

result = d.custom_vmap(f, f_batch_rule, x)
assert torch.allclose(result, 2 * x)

# autograd should let custom_vmap pass through
x = label(torch.tensor([1., 2., 3.]))
d = Autograd(Torch())
result = d.custom_vmap(f, f_batch_rule, x)
assert torch.allclose(result, x * x)
loss999 = d.sum(result, name='loss999')

grad_x, = d.grad(loss999, [x])
assert torch.allclose(grad_x, 2 * x)

# autograd should let custom_vmap pass through
x = label(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
d = Autograd(Batched(Torch(), length=2))
result = d.custom_vmap(f, f_batch_rule, x)
assert torch.allclose(result, 2 * x)
loss88 = d.sum(result, name='loss88')

grad_x, = d.grad(loss88, [x])
assert torch.allclose(grad_x, torch.full_like(x, 2))
