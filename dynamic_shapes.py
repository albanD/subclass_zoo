# -*- coding: utf-8 -*-

from typing import List, NamedTuple, Callable, Dict, Optional, Union, Any
import torch
from dataclasses import dataclass, field
import functools
import itertools
from enum import Enum

# This notebook implements dynamic shapes on top of [Simple
# Autograd](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC?usp=sharing)
# The goal is to have an easy to hack on prototype of dynamic shapes
# that you can use to explore different parts of the design space for
# dynamic shapes.

# Most of the simplest graph capture mechanisms require shape
# specialization, because they simply proceed by running an actual
# iteration of the computation with real inputs, and simply recording
# everything that occurred during the process.  This causes problems,
# however, when shapes vary across different runs (e.g., you have a
# dynamically sized input corresponding to, e.g., a string).  So
# logically, you'd like some way to record "hey, this shape isn't 1024,
# it can vary, don't make assumptions based on it happening to be 1024
# this time.)

@dataclass
class FreshSupply:
    prefix: str
    fresh: int = 0
    def __call__(self):
        r = f'{self.prefix}{self.fresh}'
        self.fresh += 1
        return r

fresh_var = FreshSupply('v')
fresh_int = FreshSupply('i')

@dataclass(frozen=True)
class Op:
    name: str
    def __str__(self):
        return self.name

# this assert should be derivable from the given preconditions in the
# program; it is inappropriate to use for "external" knowledge that is
# not derivable
int_assert_eq = Op("int_assert_eq")
var_constant = Op("var_constant")
var_add = Op("var_add")
var_mul = Op("var_mul")
var_sum = Op("var_sum")
var_expand = Op("var_expand")

def register(d, k):
    def inner(f):
        d[k] = f
    return inner

INTERP_RULES = {}

@register(INTERP_RULES, int_assert_eq)
def interp_int_assert_eq(x: int, y: int):
    assert x == y

INTERP_RULES[var_constant] = lambda *, val: val
INTERP_RULES[var_add] = lambda x, y: x + y
INTERP_RULES[var_mul] = lambda x, y: x * y
INTERP_RULES[var_sum] = lambda x: x.sum()
INTERP_RULES[var_expand] = lambda x, sizes: x.expand(sizes)

# There's a little bit of choice in the IR representation.  I chose to
# allow for complicated atoms to make the IR easier to read (no
# intermediate size allocations) at the cost of more complicated use of
# the IR.  This isn't a big deal for Z3 elaboration because we are
# fixed rank anyway.

Atom = Union[str, int, List[Union[str, int]]]

def str_atom(a: Atom) -> str:
    if isinstance(a, str):
        return a
    elif isinstance(a, int):
        return str(a)
    else:
        return f"({', '.join(str_atom(b) for b in a)})"

@dataclass
class Node:
    op: Op
    inputs: List[Atom]
    outputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        outputs_str = ', '.join(self.outputs)
        outputs_str += ' = ' if self.outputs else ''
        inputs_str = ', '.join(str_atom(a) for a in self.inputs)
        params_str = ', ' if self.inputs and self.params else ''
        params_str += ', '.join(f'{k}={v}' for k, v in self.params.items())
        return f"{outputs_str}{self.op}({inputs_str}{params_str})"

# TODO: I kind of want to stratify int/var computations, but
# it's pretty easy to pull out int comps only, and some ints
# will depend on vars (torch.unique case)
#
# I'm also obligated to represent int computations as nodes
# in the graph, as that is how compilers like XLA model it
# (not as refinement type predicates)
@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)

# We can write a little interpreter

def tuplify(outs):
    if outs is None:
        return ()
    elif isinstance(outs, tuple):
        return outs
    else:
        return (outs,)

def interpretAtom(atom: Atom, env: Dict[str, Any]):
    if isinstance(atom, str):
        return env[str]
    elif isinstance(atom, tuple):
        return tuple(interpretAtom(a) for a in atom)
    else:
        return atom

def interpretInputs(inputs: List[Atom], env: Dict[str, Any]):
    return tuple(interpretAtom(i, env) for i in inputs)

# mutates env
def interpretNode(n: Node, env: Dict[str, Any]):
    args = tuple(env[k] for k in n.inputs)
    outs = tuplify(INTERP_RULES[n.op](*args, **n.params))
    assert len(outs) == len(n.outputs)
    for k, v in zip(n.outputs, outs):
        env[k] = v

# Let's work on symbolics

# When we only deal in concrete shapes, the meaning of a shape check is
# simple: just test that the shapes are what you actually expect (this is
# what we implemented in the interpreter above).  However, when you have
# symbolic sizes, the meaning of shape checks is more murky.  For
# example, supposing x is a Tensor whose shape is symbolic, what does
# this program mean:
#
# ```
# assert x.shape[0] == 4
# ```
#
# Is this a valid program or not?  Here are two possibilities:
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
# The reasons are two fold.  First, the motivating use case for symbolic
# shapes is to aid compilers (like XLA and nvFuser) which want to avoid
# uselessly respecializing graphs where their inputs are dynamic.  It
# seems very reasonable to ask users to explicitly annotate when such
# dynamism could occur.  Second, this type of symbolic variable is
# necessary to support programs with data-dependent shapes.  Consider
# torch.unique(), a function whose output size is dependent on the data
# inside the function.  If we wish to write down the shape of this
# function without reference to the data in the tensor (which is
# typically what we want to do--we usually want to write the shapes of
# our programs in a data oblivious way), all we can really say is that
# there *exists* some (symbolic) size such that the tensor has that
# size, but no, I can't tell you what it is.  If the user then passed
# this result tensor into an operator that expects the size to actually
# be four, we would expect this to be an error.  (Now, it's *possible*
# that the user has some external knowledge that this unique() call will
# in fact always produce tensors of exactly size four, but typically
# this information would be provided out-of-band via an, e.g.,
# `output_size` argument to the function in question.)
#
# Case (2) has some useful applications; however.  If you are given an
# arbitrary model with no annotations, you can replace all of the input
# sizes with flexible sizes, run the model, and get symbolic formulas
# for what the implicit preconditions for the model are, as well as
# direct formulas for the output sizes.  If you have a spot in the
# middle of your model where you're not sure what the size should be,
# you could stick a flexible size there and have the symbolic algebra
# system tell you what the size has to be (LazyModule style).

# TODO: for simplicity, no context (hypotheses) for input shapes is currently modeled,
# but it should be.  Maybe we have assume/assert

CURRENT_GRAPH = Graph()

def record_arg(a):
    if isinstance(a, tuple):
        return tuple(record_arg(b) for b in a)
    elif isinstance(a, int):
        return a
    else:
        assert isinstance(a, (Variable, SymbolicIntNode))
        return a.name

def record_var(op, shape, *args, name=None, **kwargs):
    r = Variable(shape, name=name)
    n = Node(op, tuple(record_arg(a) for a in args), [r.name], kwargs)
    print(f'{n} : {r.shape}')
    CURRENT_GRAPH.nodes.append(n)
    return r

def record_none(op, *args, **kwargs):
    n = Node(op, tuple(record_arg(a) for a in args), [], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)

class SymbolicIntNode:
    name: str
    def __init__(self):
        self.name = fresh_int()
    def __repr__(self):
        return self.name

SymInt = Union[SymbolicIntNode, int]

def record_int(op, *args, **kwargs):
    i = SymbolicIntNode()
    n = Node(op, tuple(a.name for a in args), [i.name], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)
    return i

# if we're cool kids like FX we can use bytecode analysis to interpose
# on asserts, but since we're not we just implement it manually
def assert_int_eq(x: SymInt, y: SymInt):
    # peephole optimization
    if isinstance(x, SymbolicIntNode) and isinstance(y, SymbolicIntNode) and x.name == y.name:
        return
    if isinstance(x, int) and isinstance(y, int) and x == y:
        return
    # TODO: on the fly solve constraints to keep context small
    record_none(int_assert_eq, x, y)

# In full generality, what I'm supposed to do is run my symbolic algebra
# system whenever I do an equality test on to integer nodes and
# determine if the equality always holds, or if there are some symbolic
# inputs for which it does not hold.  If things are trivially not equal
# (e.g., rigid s asserted to be equal with 2), I want to report an
# error immediately; but I do not actually want to actually shell out to
# Z3 for these computations.
#
# To make matters worse, I want to do conditions on sizes in some
# situations; in particular, for broadcasting, I want to test if
# a size is 1.
#
# For now, we assume that if you have a SymInt, it could be anything
# (we NEVER learn anything when we do shape computations).

def definitely_one(x):
    return isinstance(x, int) and x == 1

def assert_shape_broadcast(lhs, rhs):
    # NB: if x *happens* to be one, but is not definitely one, we will
    # still reject it (even if "happens" to be the case that we would
    # broadcast here).  This is what it means for the trace to be
    # one-specialized
    r = []
    for x, y in itertools.zip_longest(reversed(lhs.shape), reversed(rhs.shape), fillvalue=1):
        if definitely_one(x):
            r.append(y)
        elif definitely_one(y):
            r.append(x)
        else:
            assert_int_eq(x, y)
            r.append(x)  # pick one arbitrarily.  TODO: immediately unify
    return tuple(reversed(r))

class Variable:
    shape: List[Union[SymInt, int]]
    name: str

    def __init__(self, shape, name: str=None):
        self.shape = shape
        self.name = name or fresh_var()

    def dim(self):
        return len(self.shape)

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes. 
    @staticmethod
    def constant(value: torch.Tensor, name: str=None):
        return record_var(var_constant, tuple(value.shape), val=value)

    def __repr__(self):
        return f"{self.name}: {self.shape}"

    # This performs a pointwise multiplication of a Variable, tracking gradients
    def __mul__(self, rhs: 'Variable') -> 'Variable':
        # defined later in the notebook
        return operator_mul(self, rhs)

    def __add__(self, rhs: 'Variable') -> 'Variable':
        return operator_add(self, rhs)

    def sum(self, name: Optional[str]=None) -> 'Variable':
        return operator_sum(self, name)

    def expand(self, sizes: List[SymInt]) -> 'Variable':
        return operator_expand(self, sizes)

class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs : List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: 'Callable[List[Variable], List[Variable]]'

gradient_tape : List[TapeEntry] = []

def reset_tape():
  gradient_tape.clear()
  global _name
  _name = 0 # reset variable names too to keep them small.

def operator_mul(self : Variable, rhs: Variable) -> Variable:
    if isinstance(rhs, float) and rhs == 1.0:
        # peephole optimization
        return self

    # define forward
    # (no broadcasting)
    shape = assert_shape_broadcast(self, rhs)
    r = record_var(var_mul, shape, self, rhs)

    # record what the inputs and outputs of the op were
    inputs = [self.name, rhs.name]
    outputs = [r.name]

    # define backprop
    def propagate(dL_doutputs: List[Variable]):
        dL_dr, = dL_doutputs

        dr_dself = rhs # partial derivative of r = self*rhs
        dr_drhs = self # partial derivative of r = self*rhs

        # chain rule propagation from outputs to inputs of multiply
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        dL_dinputs = [dL_dself, dL_drhs]
        return dL_dinputs
    # finally, we record the compute we did on the tape
    gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
    return r

def grad(L, desired_results: List[Variable]) -> List[Variable]:
    # this map holds dL/dX for all values X
    dL_d : Dict[str, Variable] = {}
    # It starts by initializing the 'seed' dL/dL, which is 1
    dL_d[L.name] = Variable.constant(torch.ones(()))
    print(f'd{L.name} ------------------------')

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
        print(f'd{L.name}_d{name}: {value.shape} = {value.name}')
    print(f'------------------------')

    return gather_grad(desired.name for desired in desired_results)

def operator_add(self : Variable, rhs: Variable) -> Variable:
    # Add follows a similar pattern to Mul, but it doesn't end up
    # capturing any variables.
    shape = assert_shape_broadcast(self, rhs)
    r = record_var(var_add, shape, self, rhs)
    def propagate(dL_doutputs: List[Variable]):
        dL_dr, = dL_doutputs
        dr_dself = 1.0
        dr_drhs = 1.0
        dL_dself = dL_dr * dr_dself
        dL_drhs = dL_dr * dr_drhs
        return [dL_dself, dL_drhs]
    gradient_tape.append(TapeEntry(inputs=[self.name, rhs.name], outputs=[r.name], propagate=propagate))
    return r

# sum is used to turn our matrices into a single scalar to get a loss.
# expand is the backward of sum, so it is added to make sure our Variable
# is closed under differentiation. Both have rules similar to mul above.

def operator_sum(self: Variable, name: Optional[str]) -> 'Variable':
    r = record_var(var_sum, (), self, name=name)
    def propagate(dL_doutputs: List[Variable]):
        dL_dr, = dL_doutputs
        size = self.shape
        return [dL_dr.expand(size)]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r


def operator_expand(self: Variable, sizes: List[SymInt]) -> 'Variable':
    assert self.dim() == 0 # only works for scalars
    r = record_var(var_expand, sizes, self, sizes)
    def propagate(dL_doutputs: List[Variable]):
        dL_dr, = dL_doutputs
        return [dL_dr.sum()]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r

def simple(a, b):
    t = a + b
    return t * b

# Let's first do the gradient computation with constants

reset_tape() # reset any compute from other cells
a_global, b_global = torch.rand(4), torch.rand(4)
a = Variable.constant(a_global, name='a')
b = Variable.constant(b_global, name='b')
loss = simple(a, b)
da, db = grad(loss, [a, b])
print("da", da)
print("db", db)

# Now let's do it again but symbolic

print("===========")
reset_tape()
s1 = SymbolicIntNode()
s2 = SymbolicIntNode()
s3 = SymbolicIntNode()
a = Variable((s1, s2))
b = Variable((s1, s3))
print("a", a)
print("b", b)
loss = simple(a, b).sum()
da, db = grad(loss, [a, b])  # expand can take symbolic sizes as argument
print("da", da)
print("db", db)
