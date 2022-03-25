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

# The purpose of this notebook explain validity hazards that apply
# to traces we collect when we are tracing code with real inputs,
# i.e., we secretly know what all the true values of our tensors/shapes
# are, and we are willing to let the user perform control flow on these
# values.  The basic problem we have to solve, is that for a tracing
# system based on overloading (as well as compiler systems that simplify
# their internals by only dealing with straight-line traces, aka tracing
# JITs--and torchdynamo falls into this category), we will have resolved
# control flow into taking a specific path, and if on a subsequent
# execution we would have taken a different path, the trace we have
# recorded is no longer valid!  So we have to detect this situation.
#
# The way this notebook will proceed, is we will implement a very simple
# set of validity tests based on "bailouts"; that is to say, we will
# sprinkle our trace with boolean tests (called bailouts) which specify
# when we had Python control flow at some point in time and relied on a
# value in our network having some specific value, so that later when we
# reexecute the trace, we can check that we would have done the same
# thing on the other path.
#
# Bailouts directly embedded in the trace are of limited use, as many
# trace compilers do not support bailout--if you think about a CUDA
# kernel fusion engine, you're just going to want to directly compile
# all of your trace into a CUDA kernel, you don't want to get halfway
# through your code and then realize "oops, I need to bail out of
# executing this kernel and do something else" (and indeed, you can't
# even do this in CUDA).  However, we can take the bailouts from our
# traces and transform them into standalone guards (thus the name of
# this notebook) which we can use to check for validity on entry.  We
# can also take a bailout and split the graph at that point, which also
# ensures validity.  Finally, if the user is willing to modify their
# code, they can make changes which make it less likely they will
# accidentally condition on a dynamic value (channeling the approach
# taken in the [Dynamic
# Shapes](https://github.com/albanD/subclass_zoo/blob/main/dynamic_shapes.ipynb)
# notebook)

# First, some imports and some utilities.

# +

import itertools
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, NoReturn

import torch


def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError("Unhandled type: {}".format(type(x).__name__))


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
fresh_bool = FreshSupply("b")
fresh_size = FreshSupply("s")


def reset():
    global CURRENT_GRAPH
    fresh_var.fresh = 0
    fresh_int.fresh = 0
    fresh_bool.fresh = 0
    fresh_size.fresh = 0
    CURRENT_GRAPH = Graph()


# -

# Once again, we need to define an IR for our traces.  Like in
# our previous notebook, we will represent each valid operation
# as an instance of the Op class, which is just a little wrapper
# around strings.


@dataclass(frozen=True)
class Op:
    name: str

    def __str__(self):
        return self.name


# Our operations are going to look slightly different this time.
# In particular, I explicitly model booleans and size tuples (in
# the other notebook, I allowed for tuples and integer literals to be
# directly expressed in the IR.)  This presentation will result in
# slightly uglier syntax, but it will easier for us to write
# interpreters and other analyses on the IR due to its uniform
# representation.

bool_bailout = Op("bool_bailout")  # b, *, expect
bool_const = Op("bool_const")  # *, val
int_eq = Op("int_eq")  # a, b
int_const = Op("int_const")  # *, val
int_placeholder = Op("int_placeholder")  # *, name
size_index = Op("size_index")  # s, *, index
size_ctor = Op("size_ctor")  # *args
var_placeholder = Op("var_placeholder")  # *, name
var_add = Op("var_add")  # a, b
var_mul = Op("var_mul")  # a, b
var_size = Op("var_size")  # a
var_expand = Op("var_expand")  # a, size

# Given a dictionary of parameters (for handling `var_placeholder`), we
# can say what the concrete interpretation of all of these operators
# should be.


def assert_(b):
    assert b


# Unlike conventional add/mul in PyTorch, these operators do not
# broadcast (we will insert explicit expands)
def prim_add(a, b):
    assert a.shape == b.shape
    return torch.add(a, b)


def prim_mul(a, b):
    assert a.shape == b.shape
    return torch.mul(a, b)


def concrete_rules(**params):
    # bool -> bool
    # int -> int
    # size -> Tuple[int, ...]
    # var -> torch.Tensor
    return {
        bool_bailout: lambda b, *, expect: assert_(b == expect),
        bool_const: lambda *, val: val,
        int_eq: lambda a, b: a == b,
        int_const: lambda *, val: val,
        int_placeholder: lambda *, name: params[name],
        size_index: lambda s, *, index: s[index],
        size_ctor: lambda *, args: args,
        var_placeholder: lambda *, name: params[name],
        var_add: prim_add,
        var_mul: prim_mul,
        var_size: lambda a: tuple(a.shape),
        var_expand: lambda a, size: a.expand(size),
    }


# Let's fill out the rest of our IR and an actual implementation
# of the interpreter (given some rules).  Unlike the dynamic shapes
# notebook, our nodes are truly in ANF (so inputs is just a list of
# string names of other nodes.)


@dataclass
class Node:
    op: Op
    inputs: List[str]
    outputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        outputs_str = ", ".join(self.outputs)
        outputs_str += " = " if self.outputs else ""
        inputs_str = ", ".join(self.inputs)
        params_str = ", " if self.inputs and self.params else ""
        params_str += ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{outputs_str}{self.op}({inputs_str}{params_str})"


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)

    def __str__(self):
        return "\n".join(str(n) for n in self.nodes)


def tuplify(outs):
    if outs is None:
        return tuple()
    else:
        return (outs,)


def interp_node(rules, n: Node, env: Dict[str, Any]):
    try:
        args = tuple(env[i] for i in n.inputs)
        outs = tuplify(rules[n.op](*args, **n.params))
        assert len(outs) == len(n.outputs)
    except Exception:
        print(f"Failed during: {n}")
        print("\n".join(f"{k} = {v}" for k, v in env.items()))
        raise
    for k, v in zip(n.outputs, outs):
        env[k] = v


def interp(graph, rules, print_env=False):
    env = {}
    for n in graph.nodes:
        interp_node(rules, n, env)
    if print_env:
        for k, v in env.items():
            print(f"{k} = {v}")


# We can show it all works with a tiny example constructing the graph
# manually by ourself.

g = Graph()
g.nodes.append(Node(var_placeholder, [], ["a"], {"name": "a"}))
g.nodes.append(Node(var_placeholder, [], ["b"], {"name": "b"}))
g.nodes.append(Node(var_add, ["a", "b"], ["r"]))

print(g)

interp(g, concrete_rules(a=torch.randn(4), b=torch.randn(4)), print_env=True)

# OK, let's write a tracer.  Like before, we will maintain a global
# graph we are tracing into, and write nodes into this graph (printing
# them as we go, because why not.)

CURRENT_GRAPH = Graph()


def record(r, op, *args, **kwargs):
    n = Node(op, [a.name for a in args], [a.name for a in tuplify(r)], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)
    return r


# The objects that we will perform tracing with, however, are going to
# operate a bit differently.  Because we are in the context of a
# concrete execution of a PyTorch program,  all of the "proxy" objects
# we are tracing with will actually have real values that correspond to
# what the untraced PyTorch program would have produced.  Our job
# is to "guard" accesses to the real values, so that we never let
# the Python program observe that an int was actually 1 unless we record
# that observance (with a `bool_bailout`).
#
# All of our proxy objects will have this structure, so we will call
# this a "Guarded" object.  Guarded objects have an actual concrete value,
# as well as a name saying how to reference them in the current graph
# trace.


class Guarded:
    name: str
    value: Any

    def __repr__(self):
        return f"{self.name}~{self.value}"


# And then we will subclass Guarded for each type we support tracing in
# our system.


class GuardedBool(Guarded):
    value: bool

    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_int()

    # The conversion to actual Python bool is when a user is actually
    # going to observe a value (ostensibly because they're doing a
    # condition on it).  So we must record a bailout here, saying that
    # on subsequent executions of this trace, the value of this boolean
    # node in the graph must match the expected value we saw initially.
    def __bool__(self):
        record(None, bool_bailout, self, expect=self.value)
        return self.value


class GuardedInt(Guarded):
    name: str
    value: int

    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_int()

    def __eq__(self, other):
        if isinstance(other, int):
            # Promote the int into a node
            other = record(GuardedInt(other), int_const, val=other)
        # Peephole optimization
        # if self.name == other.name:
        #     return record(GuardedBool(True), bool_const, val=True)
        return record(GuardedBool(self.value == other.value), int_eq, self, other)

    @staticmethod
    def placeholder(value: int, name: str = None):
        r = GuardedInt(value, name)
        return record(r, int_placeholder, name=r.name)


class GuardedSize(Guarded):
    name: str
    value: List[int]

    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_size()

    @staticmethod
    def make(value):
        return record(GuardedSize([v.value for v in value]), size_ctor, *value)

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        for s, o in zip(self, other):
            if s != o:
                return False
        return True

    def __len__(self):
        # For simplicity, we have kept everything rank specialized, so
        # we are allowed to return a raw integer here.  However, if this
        # was not OK, we could also return a GuardedInt here (and this
        # is in fact what FX does.) in fact what FX does.
        return len(self.value)

    def __getitem__(self, index: int):
        return record(GuardedInt(self.value[index]), size_index, self, index=index)


class Variable(Guarded):
    name: str
    value: torch.Tensor
    _shape = None

    @property
    def shape(self):
        # I don't want to spam the graph with repeated retrievals of the
        # size from a tensor, so we will only ever record this retrieval
        # once (and it is done lazily, on the first time you access
        # shape.)
        if self._shape is None:
            self._shape = record(GuardedSize(self.value.shape), var_size, self)
        return self._shape

    def __init__(self, value: torch.Tensor, name: str = None):
        self.value = value
        self.name = name or fresh_var()

    @staticmethod
    def placeholder(value: torch.Tensor, name: str = None):
        r = Variable(value, name)
        return record(r, var_placeholder, name=r.name)

    def dim(self):
        # choice!
        return self.value.dim()


# With this change, we can write normal looking code, including
# conditions on shapes, which we expect to be able to trace through.


def broadcast(lhs: List[GuardedInt], rhs: List[GuardedInt]) -> List[GuardedInt]:
    r = []
    for x, y in itertools.zip_longest(
        reversed(lhs.shape), reversed(rhs.shape), fillvalue=1
    ):
        if x == 1:
            r.append(y)
        elif y == 1:
            r.append(x)
        else:
            assert x == y
            r.append(x)
    return GuardedSize.make(list(reversed(r)))


# Elide expands when the sizes match
def expand(self: Variable, size: GuardedSize) -> Variable:
    if self.shape == size:
        return self
    else:
        return record(Variable(self.value.expand(size.value)), var_expand, self, size)


def add(self: Variable, rhs: Variable) -> Variable:
    shape = broadcast(self, rhs)
    self_expanded = expand(self, shape)
    rhs_expanded = expand(rhs, shape)
    return record(
        Variable(self_expanded.value + rhs_expanded.value),
        var_add,
        self_expanded,
        rhs_expanded,
    )


# Let's take a look at the trace produced by this code, and see
# in particular what bailouts got produced.

torch.manual_seed(0)
a = Variable.placeholder(torch.randn(4), "a")
b = Variable.placeholder(torch.randn(4), "b")
v0 = add(a, b)

# There is a lot of code generated and a lot of bailouts, and it's
# kind of hard to see what is going on.  The first three bailouts result
# from the broadcasting test: we have to check if each of the sizes
# are one (that's two bailouts), and then we check if the sizes are
# equal (the third bailout).  Then, when we run expand on the shape,
# there is another equality test between the computed broadcasted
# shape and the size, which results in the last two bailouts.
#
# The graph here is quite ugly, so let's try to clean it up a little.
# Because this graph has no data-dependent control flow, we can
# recompute the bailouts in terms of integer computations ONLY, by
# running a slightly different interpreter which maps tensors to
# their (possibly dynamic) shapes, but otherwise works the same way
# as before (we'll also assume that everything is well typed, which
# it is, assuming the bailouts in the original program are sufficient).


def shape_rules(**params):
    # bool -> GuardedBool
    # int -> GuardedInt
    # size -> GuardedSize
    # var -> GuardedSize !!!
    return {
        bool_bailout: lambda b, *, expect: record(
            None, bool_bailout, b, expect=b.value
        ),
        bool_const: lambda *, val: val,
        int_eq: lambda a, b: a == b,
        int_const: lambda *, val: val,
        int_placeholder: lambda *, name: params[name],
        size_index: lambda s, *, index: s[index],
        size_ctor: lambda *args: args,
        var_placeholder: lambda *, name: params[name],
        var_add: lambda a, b: a,
        var_mul: lambda a, b: a,
        var_size: lambda a: a,
        var_expand: lambda a, size: size,
    }


# Let's save our current trace, and reset the context for the new
# trace we are construct by interpret our original trace with
# the shape rules.

graph = CURRENT_GRAPH
reset()

# Now we can see that we get a graph with only integer/bool operations
# in it!

interp(
    graph,
    shape_rules(
        a=[GuardedInt.placeholder(4, "a0")], b=[GuardedInt.placeholder(4, "b0")]
    ),
)

# We can run this graph, asking if a new set of concrete sizes
# is valid or not.

try:
    interp(CURRENT_GRAPH, concrete_rules(a0=4, b0=8))
except Exception:
    traceback.print_exc()

# We can also write a little printer for our trace to say what our
# guards should be.


def print_rules():
    return {
        bool_bailout: lambda b, *, expect: print(
            f"assert {b}" if expect else f"assert not {b}"
        ),
        bool_const: lambda *, val: str(val),
        int_eq: lambda a, b: f"({a} == {b})",
        int_const: lambda *, val: str(val),
        int_placeholder: lambda *, name: name,
    }


interp(CURRENT_GRAPH, print_rules())

# Obviously there is some redundancy, but you can write a little
# optimizer to clean it up, or send it to your favorite symbolic
# reasoning engine.  For example, you can see that the constraints
# here are only equalities on one, and equalities between items;
# so we could easily use unification to get these into canonical form.
#
# TODO: Relate this to symbolic dynamic sizes which prevent constraints
# from being generated.
