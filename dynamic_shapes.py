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

# This notebook implements rank specialized dynamic shapes on [Simple
# Autograd](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC?usp=sharing)
# The goal is to have an easy to hack on prototype of dynamic shapes
# that you can use to explore different parts of the design space for
# dynamic shapes.
#
# Most of the simplest graph capture mechanisms require shape
# specialization, because they simply proceed by running an actual
# iteration of the computation with real inputs, and simply recording
# everything that occurred during the process.  This causes problems,
# however, when shapes vary across different runs (e.g., you have a
# dynamically sized input or you use data-dependent operators like
# nonzero/unique).  So logically, you'd like some way to record "hey,
# this shape isn't 1024, it can vary, don't make assumptions based on it
# happening to be 1024 this time.)

# +
import functools
import itertools
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch

torch.manual_seed(0)
# -

# To get started, we define some utility functions the autograd
# implementation.  FreshSupply lets us generate fresh names for
# variables and symbolic integers in our program (for clarity,
# we give them separate prefixes: v for variable, i for integer).
# `gradient_tape` contains the global autograd tape for our
# program; for more details, read the exposition in [Simple
# Autograd](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC?usp=sharing).


# +
@dataclass
class FreshSupply:
    prefix: str
    fresh: int = 0

    def __call__(self):
        r = f"{self.prefix}{self.fresh}"
        self.fresh += 1
        return r


fresh_var = FreshSupply("v")
fresh_int = FreshSupply("i")


class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs: List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: "Callable[List[Variable], List[Variable]]"


gradient_tape: List[TapeEntry] = []


def reset():
    gradient_tape.clear()
    fresh_var.fresh = 0
    fresh_int.fresh = 0
    CURRENT_GRAPH.nodes.clear()


# -

# The raison d'etre of dynamic shapes is for graph capture, and so to
# start we will define a little IR representing sequences of operations
# that we will be recording.  To avoid string confusion, we'll use
# a little wrapper class Op to represent each operation; we'll give
# meaning to these objects later by defining interpretation rules for
# them.

# +


@dataclass(frozen=True)
class Op:
    name: str

    def __str__(self):
        return self.name


# In this example, we only have asserts over integers, but you could
# imagine other integer operations, e.g., addition, for handling
# operators like torch.cat
int_assert_eq = Op("int_assert_eq")
var_constant = Op("var_constant")
var_add = Op("var_add")
var_mul = Op("var_mul")
var_sum = Op("var_sum")
var_expand = Op("var_expand")
# Unlike nonzero, this returns both the number of nonzero elements, as
# well as the return tensor
var_nonzero_impl = Op("var_nonzero_impl")
var_index = Op("var_index")  # aka x[i]
var_index_backward = Op("var_index_backward")
var_squeeze = Op("squeeze")
var_unsqueeze = Op("unsqueeze")

# -

# Let's write a little interpreter for these these operations.  Their
# implementations will proceed as you might expect.  Since there are a
# lot of operators, we'll put all of the functions for handling these
# operations in an INTERP_RULES dictionary.  The interpreter ops
# distinguish between args and kwargs; args are dynamic data that is
# allowed to depend on other operations in our IR, whereas kwargs
# is static data which never depends on other computation.
#
# Note that operations on integers are also represented in the graph!
# In some graph representations, integers are represented as scalar
# tensors, but for clarity in this presentation they are represented by
# a separate type.

# +


def register(d, k):
    def inner(f):
        d[k] = f

    return inner


INTERP_RULES = {}


@register(INTERP_RULES, int_assert_eq)
def interp_int_assert_eq(x: int, y: int):
    assert x == y


# unlike nonzero, this also returns the symbolic shape
# because this is an "existential telescope" the fresh shape gotta come
# first
@register(INTERP_RULES, var_nonzero_impl)
def interp_var_nonzero_impl(x: torch.Tensor):
    r = torch.nonzero(x)
    return r.shape[0], r


INTERP_RULES[var_index] = lambda t, i: t[i]
# NB: this is inefficient: t's data doesn't to be retained to allocate
# the zeros, only the dtype (actually technically inferrable from g) and size
INTERP_RULES[var_index_backward] = lambda t, i, g: torch.zeros_like(t).index_put(
    (i,), g, accumulate=True
)
INTERP_RULES[var_constant] = lambda *, val: val
INTERP_RULES[var_add] = lambda x, y: x + y
INTERP_RULES[var_mul] = lambda x, y: x * y
INTERP_RULES[var_sum] = lambda x: x.sum()
INTERP_RULES[var_expand] = lambda x, sizes: x.expand(sizes)
INTERP_RULES[var_squeeze] = lambda x, *, dim: x.squeeze(dim)
INTERP_RULES[var_unsqueeze] = lambda x, *, dim: x.unsqueeze(dim)
# -

# Let's actually define IR nodes that make use of these operations.
# In many IRs, you only allow variables as arguments (so called
# "administrative normal form"); but in this treatment, I chose to also
# allow tuples in atoms to make the IR easier to read (as we can just
# directly specify shapes as arguments, rather than having to first
# allocate a tuple and then pass it the function).  There is no
# need for first class tuples because we are dealing with *rank
# specialized* dynamic shapes, so the size of tuples is always known
# ahead of time.  Similarly, if an integer argument in a shape is
# known statically ahead of time, we'll let you just inline it into
# the argument site.  We will call these structures atoms.

# +
Atom = Union[str, int, List[Union[str, int]]]


def str_atom(a: Atom) -> str:
    if isinstance(a, str):
        return a
    elif isinstance(a, int):
        return str(a)
    else:
        return f"({', '.join(str_atom(b) for b in a)})"


# -

# Our node looks fairly similar to an FX node: given some list of input
# atoms and a dictionary of arbitrary extra (static) parameters, run
# an operator on it and bind the results to outputs.


@dataclass
class Node:
    op: Op
    inputs: List[Atom]
    outputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        outputs_str = ", ".join(self.outputs)
        outputs_str += " = " if self.outputs else ""
        inputs_str = ", ".join(str_atom(a) for a in self.inputs)
        params_str = ", " if self.inputs and self.params else ""
        params_str += ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{outputs_str}{self.op}({inputs_str}{params_str})"


# A graph is simply a list of nodes.  In a real application we may also
# need some provision for graph-level inputs/outputs, but they aren't
# necessary for these example so they are omitted.


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)


# Now, we can finish our interpreter on graphs.

# +


def tuplify(outs):
    if outs is None:
        return ()
    elif isinstance(outs, tuple):
        return outs
    else:
        return (outs,)


def interp_atom(atom: Atom, env: Dict[str, Any]):
    if isinstance(atom, str):
        return env[atom]
    elif isinstance(atom, tuple):
        return tuple(interp_atom(a, env) for a in atom)
    else:
        return atom


def interp_inputs(inputs: List[Atom], env: Dict[str, Any]):
    return tuple(interp_atom(i, env) for i in inputs)


# Mutates the environment, storing the results into env
def interp_node(n: Node, env: Dict[str, Any]):
    args = tuple(interp_atom(i, env) for i in n.inputs)
    try:
        outs = tuplify(INTERP_RULES[n.op](*args, **n.params))
    except Exception:
        print(f"Failed during: {n}")
        raise
    assert len(outs) == len(n.outputs)
    for k, v in zip(n.outputs, outs):
        env[k] = v


# -

# To put it all together, we have a final `interp_graph` function,
# which takes in initial values for variables and then runs the
# graph recorded at `CURRENT_GRAPH` to compute some output variables
# (which it then prints.)


def interp_graph(init: Dict[Union["Variable", "SymbolicIntNode"], Any], **outs):
    env = {k.name: v for k, v in init.items()}
    for n in CURRENT_GRAPH.nodes:
        interp_node(n, env)
    for k, v in outs.items():
        print(f"{k} = {env[v.name]}")


# Phew, that's it for the IR representation.  Let's actually generate
# some graphs!  We'll maintain a global current graph which we record
# into.

CURRENT_GRAPH = Graph()

# Like FX, XLA and LazyTensor, we will generate graphs by maintaining
# symbolic "proxy" objects (Variable, SymbolicIntNode) which act
# like normal tensors/integers but don't actually contain any data
# and record the operations that occur on them.  Our symbolic integers
# in this example have a very impoverished interface; in fact they
# support no operations at all, they simply have a name corresponding
# to their name in the graph.

# +


class SymbolicIntNode:
    name: str

    def __init__(self, name=None):
        self.name = name or fresh_int()

    def __repr__(self):
        return self.name


class Variable:
    shape: List[Union["SymInt", int]]
    name: str
    dtype: torch.dtype

    def __init__(self, shape, dtype: torch.dtype, *, name: str = None):
        self.shape = shape
        self.dtype = dtype
        self.name = name or fresh_var()

    def dim(self):
        return len(self.shape)

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value: torch.Tensor, name: str = None):
        return record_var(var_constant, tuple(value.shape), value.dtype, val=value)

    def __repr__(self):
        return f"{self.name}: {self.shape}"

    # Most of the actual implementations of these operators will be
    # given later.

    def __mul__(self, rhs: "Variable") -> "Variable":
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: "Variable") -> "Variable":
        return operator_add(self, rhs)

    def sum(self, name: Optional[str] = None) -> "Variable":
        return operator_sum(self, name)

    def expand(self, sizes: List["SymInt"]) -> "Variable":
        return operator_expand(self, sizes)

    def nonzero(self) -> "Variable":
        return operator_nonzero(self)

    def squeeze(self, dim: int) -> "Variable":
        return operator_squeeze(self, dim)

    def unsqueeze(self, dim: int) -> "Variable":
        return operator_unsqueeze(self, dim)

    def zeros_like(self) -> "Variable":
        return zeros_like(self)

    def __getitem__(self, index) -> "Variable":
        return operator_index(self, index)

    def index_backward(self, index, grad_output) -> "Variable":
        return operator_index_backward(self, index, grad_output)


# -

# We allow integers to be specialized (indeed, in lazy tensor, most
# integers in networks will be statically known), so you may be
# dealing with a literal integer or a symbolic integer node.  SymInt
# captures these two possibilities.

SymInt = Union[SymbolicIntNode, int]

# To record a graph node involving integers (e.g., `int_assert_eq`),
# we simply take in the input Variables/SymInts, extract out their
# names, and record an operator to the current graph for the operation
# we tried to do.  Furthermore, if the operation returns a result
# (e.g., an int or a tensor), we have to construct a new proxy object
# representing the result.  We provide helper functions for doing this
# with operations that return nothing, a single int, or a single
# variable (note that the to return a new variable proxy, we must
# provide a (possibly symbolic but definitely rank specialized) shape
# and a (concrete) dtype of the result.  (dtypes are concrete because
# we don't support symbolic dtype!  That is a much rarer thing to want
# to do inside of a network.)

# +


def record_arg(a):
    if isinstance(a, tuple):
        return tuple(record_arg(b) for b in a)
    elif isinstance(a, int):
        return a
    else:
        assert isinstance(a, (Variable, SymbolicIntNode))
        return a.name


def record_none(op, *args, **kwargs):
    n = Node(op, tuple(record_arg(a) for a in args), [], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)


def record_int(op, *args, **kwargs):
    i = SymbolicIntNode()
    n = Node(op, tuple(a.name for a in args), [i.name], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)
    return i


def record_var(
    op, shape: Tuple[SymInt, ...], dtype: torch.dtype, *args, name=None, **kwargs
):
    r = Variable(shape, dtype, name=name)
    n = Node(op, tuple(record_arg(a) for a in args), [r.name], kwargs)
    print(f"{n} : {r.shape}")
    CURRENT_GRAPH.nodes.append(n)
    return r


# -

# We our now ready to define our first operation: `int_assert_eq`.
# Although we peephole a few cases where this assertion is vacuously
# true, in general we don't know if two symbolic integers are actually
# equal or not, so we just record an assertion in the graph to be
# verified later at runtime.


def assert_int_eq(x: SymInt, y: SymInt):
    # peephole optimization
    if (
        isinstance(x, SymbolicIntNode)
        and isinstance(y, SymbolicIntNode)
        and x.name == y.name
    ):
        return
    if isinstance(x, int) and isinstance(y, int) and x == y:
        return
    record_none(int_assert_eq, x, y)


# It is not implemented in this notebook, but what if you wanted to
# immediately check this assertion *while tracing* (instead of waiting
# for concrete inputs/sizes).  There is now an interesting choice you have
# to make.  In particular, suppose x is a Tensor whose shape is
# symbolic; what does this program mean?
#
# ```
# assert x.shape[0] == 4
# ```
#
# Should you error during tracing or not?  There are two possibilities:
#
# 1. **The program is invalid.**  In this interpretation, the
#    shape of x being symbolic is a claim that this program should
#    work for any choice of x, and now we are trying to **check** that
#    this is actually true.  When this assert occurs, we say, "The user
#    told us x.shape[0] could be anything, but now we see that if
#    x.shape[0] is not 4 this assert will fail; there's a bug and
#    we should report an error."  This is the sort of thing you would
#    do if you were given a model with specific sizes annotated as
#    "dynamic" and then were trying to trace the model under this
#    assumption.  We call these symbolic sizes **rigid**, because
#    they never change as we execute a program.
#
#    Note: it's not strictly required to let a size range over all
#    integers; for example, a user could specify preconditions for their
#    model input shapes (e.g., "x.shape[0] must be even") which could
#    then be used to show that asserts on those shapes must be true.  In
#    the degenerate case, the preconditions specify concrete values for
#    every shape in the program, making this equivalent to the concrete
#    shapes case.  In XLA, the information about sizes having upper
#    bounds serves a similar role.
#
# 2. **The program is valid.**  In this interpretation, the shape
#    of x being symbolic just means we don't know anything about the
#    preconditions/postconditions of our program, and we are trying to
#    **infer** them from program.  When this assert occurs, we say, "Aha!
#    Now we know that x.shape[0] is 4" and can use that fact in later
#    analysis (e.g., to find a contradiction with a later assert, which
#    would indicate that there is a bug in the program).  This is the
#    sort of thing you would do if you were given a model with no
#    input/output shape annotations and were trying to infer what the top
#    level annotations should be.  We call these symbolic sizes
#    **flexible**, because their nature will change based on the program
#    we run them on.
#
# If you told me I could only support one case, I would pick case (1).
# To see why, consider torch.nonzero(), a function whose output size is
# dependent on the data inside the function.  If we wish to write down
# the shape of this function without reference to the data in the tensor
# (which is typically what we want to do--we usually want to write the
# shapes of our programs in a data oblivious way), all we can really say
# is that there *exists* some (symbolic) size such that the tensor has
# that size, but no, I can't tell you what it is.  If the user then
# passed this result tensor into an operator that expects the size to
# actually be four, we would expect this to be an error.  (Now, it's
# possible that there *accidentally* were four nonzero elements, but
# I wouldn't bet on it!)
#
# Case (2) has some useful applications; however.  If you are given an
# arbitrary model with no annotations, you can replace all of the input
# sizes with flexible sizes, run the model, and get symbolic formulas
# for what the implicit preconditions for the model are, as well as
# direct formulas for the output sizes.  If you have a spot in the
# middle of your model where you're not sure what the size should be,
# you could stick a flexible size there and have the symbolic algebra
# system tell you what the size has to be (LazyModule style).
#
# In fact, case (1) and case (2) can coexist in the same model.  There
# are a few ways to do it: you could distinguish between rigid/flexible
# symbolic variables and have asserts work differently depending on the
# variable in question/you could also introduce two different types
# of asserts (one which says "this assert is derivable from the
# shape preconditions of this model" and another which says "this assert
# represents external knowledge that I have about the model, and you can
# now use this for further reasoning.")  They're not implemented here,
# but it is a good exercise to try.


# We our now ready to implement pointwise multiplication.  In this
# notebook, we have chosen to faithfully replicate *broadcasting*
# semantics, which means that adding two tensors with two different
# sizes at a dimension is OK if one of the sizes is one.
#
# To do this, we need a way of taking a (possibly) symbolic integer
# and checking (at trace time) if it is one.  For dynamic sizes,
# there is no way to generically do this, because we haven't run
# the program, we have no idea what the size will be!  So the simplest
# possible choice is to only report a SymInt as one if it is
# *specialized* to be one (e.g., it is literally a one integer.)


def definitely_one(x: SymInt) -> bool:
    return isinstance(x, int) and x == 1


# This reasoning is not complete.  Suppose x has size (s0,) and y has
# size (s1,), and we have:
#
# ```
# assert x.shape[0] == 1
# return x + y
# ```
#
# Does broadcasting occur on this addition?  A user might reasonably
# expect that after this assertion, surely broadcasting should occur.
# But if we are relying on x.shape[0] to *literally* be one (as we
# are in the implementation of `definitely_one` above), we won't figure
# it out!  It's not too difficult to figure it out though: in
# particular, if you implemented unification that would suffice.
#
# The trouble is, it's not well specified *how much* symbolic reasoning
# we should be willing to do.  It seems that unification is necessary,
# lest obvious instances of transitivity don't work (x == 1, y == x, z
# == y, operations on z should broadcast).  On the flip side, we
# shouldn't be in the business of proving arbitrary mathematical
# theorems (or even shelling to Z3) to figure out if a value is always
# one in some context.  But what about a ResNet style architecture,
# where the output size of layers gets reduced and reduced until it
# hits 1; should we be able to infer that the output of a ResNet in this
# case is size 1 and eligible for broadcasting?  (Does this even matter,
# since the input sizes in such networks are typically static and we
# wouldn't have a dynamic shape in this case anyway?)  To definitively
# answer these questions would require an analysis of broadcasting usage
# in the wild (or perhaps an analysis of networks with dynamic sizes.)
#
# There is sense in which unification is "good enough", however.  If our
# symbolic reasoning is insufficient for a user, they can always add an
# assert that a shape in question is one to force broadcasting to occur
# in that case.  Then, the problem reduces to letting a user know
# that this is what they ought to do; if two tensors don't broadcast
# with each other, we may just require their sizes to be the same; but
# it might not be obvious (without the help of a solver like Z3) that
# two symbolic sizes are different.  If we lowered to an IR with
# non-broadcasting operations, this will manifest at runtime where we'll
# say "Couldn't add tensors with sizes 1 and N" (even though the surface
# language supported broadcasting).  So you want to help the user out
# here with a better error message, in that case.

# Anyway, now that we have `definitely_one`, we can implement an
# assertion that two shapes are broadcastable (returning the
# possible symbolic shape out the end).  This broadcasting is more
# conservative than actual PyTorch, because if a shape is dynamic
# it will reject it (even if at runtime it happened to be one and
# therefore the broacasting woudl have succeeded).


def assert_shape_broadcast(lhs, rhs):
    r = []
    for x, y in itertools.zip_longest(
        reversed(lhs.shape), reversed(rhs.shape), fillvalue=1
    ):
        if definitely_one(x):
            r.append(y)
        elif definitely_one(y):
            r.append(x)
        else:
            assert_int_eq(x, y)
            r.append(x)  # pick one arbitrarily
    return tuple(reversed(r))


# Finally, we can implement multplication!


def operator_mul(self: Variable, rhs: Variable) -> Variable:
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # Broadcast the two shapes together, getting the result shape
    shape = assert_shape_broadcast(self, rhs)
    # We didn't implement type promotion, so just assert that the
    # inputs are the same dtype.
    assert self.dtype == rhs.dtype
    # Record the operation into the graph
    r = record_var(var_mul, shape, self.dtype, self, rhs)

    # Record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # Define backprop.  This closes over self and rhs, which are
    # necessary to define the backwards rule.
    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs

        dr_dself = rhs  # partial derivative of r = self*rhs
        dr_drhs = self  # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs

    # Finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r


# The implementation of gradient computation works exactly the same
# way it did in Simple Grad.


def grad(L, desired_results: List[Variable]) -> List[Variable]:
    # this map holds dL/dX for all values X
    dL_d: Dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable.constant(torch.ones(()))
    print(f"d{L.name} ------------------------")

    # look up dL_dentries. If a variable is never used to compute the loss,
    # we consider its gradient None, see the note below about zeros for more information.
    def gather_grad(entries: List[str]):
        return [dL_d[entry] if entry in dL_d else None for entry in entries]

    # propagate the gradient information backward
    for entry in reversed(gradient_tape):
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
                dL_d[input] = dL_d[input] + dL_dinput

    # print some information to understand the values of each intermediate
    for name, value in dL_d.items():
        print(f"d{L.name}_d{name}: {value.shape} = {value.name}")
    print(f"------------------------")

    return gather_grad(desired.name for desired in desired_results)


# Add, sum and expand look identical to their versions in Simple Grad.
# One thing to note, however: expand takes sizes as input, and those
# sizes can be symbolic!


def operator_add(self: Variable, rhs: Variable) -> Variable:
    shape = assert_shape_broadcast(self, rhs)
    assert self.dtype == rhs.dtype  # no type promotion
    r = record_var(var_add, shape, self.dtype, self, rhs)

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]

    gradient_tape.append(
        TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_sum(self: Variable, name: Optional[str]) -> "Variable":
    r = record_var(var_sum, (), self.dtype, self, name=name)

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        size = self.shape
        return [dL_dr.expand(size)]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


# NB: -1 sizes was not implemented
def operator_expand(self: Variable, sizes: List[SymInt]) -> "Variable":
    assert self.dim() == 0  # only works for scalars
    r = record_var(var_expand, sizes, self.dtype, self, sizes)

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        return [dL_dr.sum()]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


# With these operators, we can reprise the simple add-multiply example
# from the original Simple Grad, but this time first symbolically
# tracing the graph, and then executing it after the fact.

# +


def simple(a, b):
    t = a + b
    return t * b


reset()  # reset any compute from other cells
# We do no computation in this part, we're just tracing!
a = Variable((4,), dtype=torch.float, name="a")
b = Variable((4,), dtype=torch.float, name="b")
loss = simple(a, b)
da, db = grad(loss, [a, b])
# Setting a and b to random tensors, run the interpreted graph
# and print out the result of da and db.
va = torch.randn(4)
vb = torch.randn(4)
interp_graph({a: va, b: vb}, da=da, db=db)

# -

# In the above example, we still had variables with completely concrete
# sizes.  We can also replace all of the input sizes with dynamic sizes,
# and get a symbolic trace.  Note that in this example we gave a and
# b distinct symbolic sizes, but actually the network requires them to
# be the same and you can see the generated asserts.

reset()
s1 = SymbolicIntNode(name="s1")
s2 = SymbolicIntNode(name="s2")
a = Variable((s1,), dtype=torch.float)
b = Variable((s2,), dtype=torch.float)
loss = simple(a, b).sum()
da, db = grad(loss, [a, b])  # expand can take symbolic sizes as argument
interp_graph({s1: 4, s2: 4, a: va, b: vb}, da=da, db=db)

# In fact, with our simple interpreter, we will fail the assert EVEN if
# the original source program would have worked by broadcasting a
# one-sized dimension.  This might be undesirable, but there are some
# other ways to solve the problem:
#
#   - Arguably, the symbolic integer assignments here are wrong:
#     for the internal addition/multiplication to be non-broadcasting,
#     the sizes of the two inputs have to be the same.  So we could
#     require the user to give a more precise set of preconditions.
#
#   - Alternately, if we are operating as a lazy tensor, the
#     preconditions just say when we need to invalidate an old trace
#     (and build a new trace with the assumption that it is
#     broadcasting).

try:
    interp_graph({s1: 1, s2: 4, a: va, b: vb}, da=da, db=db)
except Exception:
    traceback.print_exc()

# One of the motivating operators for dynamic shapes is nonzero (which
# is used in MaskRCNN, among other models).  It's trace implementation
# embodies the standard technique for dealing with unknown output sizes:
# allocate a fresh SymbolicIntNode and put *that* in the size of the
# returned tensor!
#
# There is a slight twist: inside the graph, it returns both a symbolic
# int (the number of nonzero elements) as well as the actual result
# tensor (containing the indices of the nonzero elements).  This is
# necessary to ensure that the symbolic int is in scope for later
# operations!  An alternative way to model this that was taken by XLA is
# to return simply the tensor, and then represent the symbolic int as a
# "get size" operation in the graph.  However, I like this orientation
# better, as it seems more logical (the variable depends on the symbolic
# integer, not the other way around).


def operator_nonzero(self: Variable) -> "Variable":
    s = SymbolicIntNode()
    r = Variable((s, self.dim()), torch.long)
    n = Node(var_nonzero_impl, (self.name,), [s.name, r.name])
    print(f"{n} : {r.shape}")
    CURRENT_GRAPH.nodes.append(n)
    return r


# Nonzero isn't very interesting on its own, so I also included
# implementations of advanced indexing which can make use of the
# LongTensor returned by nonzero (as well as a squeeze, as you need to
# chop off the last dimension returned by nonzero to get the output
# compatible with indexing).  In MaskRCNN, the general idea of the code
# is that you get the nonzero indexes, index them out of the tensor,
# and then compute your loss only on those entries (because they're the
# boxes that MaskRCNN actually selected for recognition!)


def operator_index(self: Variable, index: Variable) -> "Variable":
    assert isinstance(index, Variable)  # no slices support
    assert index.dtype == torch.long  # integer indexing only
    assert index.dim() == 1  # 1D index only
    r = record_var(
        var_index, (index.shape[0],) + self.shape[1:], self.dtype, self, index
    )

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        return [self.index_backward(index, dL_dr)]

    # NB: index not recorded on tape as it is nondifferentiable
    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_index_backward(
    self: Variable, index: Variable, grad_output: Variable
) -> "Variable":
    assert isinstance(index, Variable)
    assert index.dtype == torch.long  # integer indexing only
    assert index.dim() == 1  # 1D index only
    assert_int_eq(grad_output.shape[0], index.shape[0])
    # no broadcasting
    for i in range(1, len(self.shape)):
        assert_int_eq(self.shape[i], grad_output.shape[i])
    r = record_var(var_index_backward, self.shape, self.dtype, self, index, grad_output)

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_doutputs
        return [dL_dr[index]]

    # NB: self and index not recorded as they are nondifferentiable
    gradient_tape.append(
        TapeEntry(inputs=[grad_output.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_squeeze(self: Variable, dim: int) -> "Variable":
    # Technically, squeeze is supposed to noop if the dimension isn't
    # size 1.  But if the shape in question is dynamic we don't
    # know if it is one or not.  For now, we just assert that it has to
    # be size 1 and reduce, but technically we should use definitely_one
    # to go between behavior
    if not definitely_one(self.shape[dim]):
        raise RuntimeError("cannot squeeze on dynamic dimension")
    r = record_var(
        var_squeeze,
        self.shape[0:dim] + self.shape[dim + 1 :],
        self.dtype,
        self,
        dim=dim,
    )

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_outputs
        # NB: This is only the backwards if a squeeze actually occurs
        return [dL_dr.unsqueeze(dim)]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


def operator_unsqueeze():
    assert_int_eq(self.shape[dim], 1)
    r = record_var(
        var_unsqueeze,
        self.shape[0:dim] + (1,) + self.shape[dim:],
        self.dtype,
        self,
        dim=dim,
    )

    def propagate(dL_doutputs: List[Variable]):
        (dL_dr,) = dL_outputs
        return [dL_dr.squeeze(dim)]

    gradient_tape.append(
        TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate)
    )
    return r


# As a warmup, let's show that indexing works.

reset()
a = Variable((2, 3), dtype=torch.float)
i = Variable((4,), dtype=torch.long)
loss = a[i].sum()
(da,) = grad(loss, [a])
interp_graph({a: torch.randn(2, 3), b: torch.tensor([0, 0, 0, 1])}, da=da)


# Now, let's do a nontrivial symbolic case, where we index based on the
# result of nonzero

reset()
a = Variable((6,), dtype=torch.float)
i = a.nonzero().squeeze(1)
loss = a[i].sum(name="L0")
(da,) = grad(loss, [a])
interp_graph({a: torch.clamp(torch.randn(6), min=0)}, da=da)

# I didn't finish everything that I wanted to in this prototype.  Here
# are more things that could be done:
#
#   - We have a relatively complicated design for non-refcounted SymInt
#     in C++.  This design could be implemented here to get a better
#     understanding of how explicit reference counting affects the
#     user experience.
#
#   - Right now, the symbolic traces represent add/mul as their
#     broadcasting versions.  It is easy to tweak it using
#     `definitely_one` to explicitly represent the broadcasting
#     using an expand first.
#
#   - XLA's dynamic shape support also tracks upper bounds for all
#     symbolic sizes.  Incorporate support for that here.
#
#   - Incorporate some level of symbolic reasoning, improving the
#     precision of `definitely_one` or letting us check the validity
#     asserts while tracing (and not only just at runtime).  A
#     good start would be unification, but backending to Z3 would also
#     be interesting.
#
#   - Add some operators with nontrivial input/output shape
#     relationships (e.g., addition, division, etc.)
