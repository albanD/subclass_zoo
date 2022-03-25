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


import itertools
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import (
    Any,
    Dict,
    List,
    NoReturn,
    Optional,
)

import torch

torch.manual_seed(0)


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
    fresh_var.fresh = 0
    fresh_int.fresh = 0
    fresh_bool.fresh = 0
    fresh_size.fresh = 0
    CURRENT_GRAPH.nodes.clear()


@dataclass(frozen=True)
class Op:
    name: str

    def __str__(self):
        return self.name


bool_bailout = Op("bool_bailout")  # b, *, expect
int_eq = Op("int_eq")  # a, b
int_const = Op("int_const")  # *, val
size_index = Op("size_index")  # s, *, index
size_ctor = Op("size_ctor")  # *args
var_placeholder = Op("var_placeholder")  # *, name
var_add = Op("var_add")  # a, b
var_mul = Op("var_mul")  # a, b
var_size = Op("var_size")  # a
var_expand = Op("var_expand")  # a, size

# if you want to do on the fly


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


CURRENT_GRAPH = Graph()


def tuplify(outs):
    if outs is None:
        return tuple()
    else:
        return (outs,)


def record(r, op, *args, **kwargs):
    n = Node(op, [a.name for a in args], [a.name for a in tuplify(r)], kwargs)
    print(n)
    CURRENT_GRAPH.nodes.append(n)
    return r


class Guarded:
    name: str
    value: Any

    def __repr__(self):
        return f"{self.name}~{self.value}"


class GuardedBool(Guarded):
    value: bool

    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_int()

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
            # promote the int into a node
            other = record(GuardedInt(other), int_const, val=other)
        return record(GuardedBool(self.value == other.value), int_eq, self, other)


class GuardedSize(Guarded):
    name: str
    value: List[int]

    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_size()

    @staticmethod
    def make(value):
        return record(GuardedSize([v.value for v in value]), size_ctor, *value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, index: int):
        return record(GuardedInt(self.value[index]), size_index, self, index=index)


class Variable(Guarded):
    name: str
    value: torch.Tensor
    _shape = None

    @property
    def shape(self):
        # TODO: lazy recording
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


# bad version
def broadcast(lhs: List[int], rhs: List[int]) -> List[int]:
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


def expand(self: Variable, size: GuardedSize) -> Variable:
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


# here we have some bailouts

a = Variable.placeholder(torch.randn(4), "a")
b = Variable.placeholder(torch.randn(4), "b")
v0 = add(a, b)

# look at the bailouts.  They are not very good.
# - trace would get invalidated if any dimension happened to be 1,
# even if both symbolic dims are ok
# - on the fly ness


# Mutates the environment, storing the results into env
def interp_node(rules, n: Node, env: Dict[str, Any]):
    args = tuple(env[i] for i in n.inputs)
    try:
        outs = tuplify(rules[n.op](*args, **n.params))
        assert len(outs) == len(n.outputs)
    except Exception:
        print(f"Failed during: {n}")
        print("\n".join(f"{k} = {v}" for k, v in env.items()))
        raise
    for k, v in zip(n.outputs, outs):
        env[k] = v


def interp_graph(graph, rules, init: Optional[Dict[str, Any]] = None, print_env=False):
    if init:
        env = {k: v for k, v in init.items()}
    else:
        env = {}
    for n in graph.nodes:
        interp_node(rules, n, env)
    if print_env:
        for k, v in env.items():
            print(f"{k} = {v}")


def assert_(b):
    assert b


interp_rules = {
    bool_bailout: lambda b, *, expect: assert_(b == expect),
    int_eq: lambda a, b: a == b,
    int_const: lambda *, val: val,
    # TODO: do the rest
}

# domain: sets of placeholder names
dependency_rules = {
    bool_bailout: lambda b, *, expect: print(b),
    int_eq: lambda a, b: a | b,
    int_const: lambda *, val: set(),
    size_index: lambda s, *, index: s,
    size_ctor: lambda *args: reduce(operator.ior, args),
    var_placeholder: lambda *, name: {name},
    var_add: lambda a, b: a | b,
    var_mul: lambda a, b: a | b,
    var_size: lambda a: a,
    var_expand: lambda a, b: b,
}
interp_graph(CURRENT_GRAPH, dependency_rules)


# 1-5, 5-1, 5-5 cases easy
# 1-1 case need to guess (broadcast or equal?)  guess equal.
#
# 1-specialize says: preferentially resolve on things being 1 first
#
# pool example (division by 2)
#
# bailout vs split


# we can compute the residual bailouts

"""
def broadcast(a: List[int], b: List[int]):
"""


env = {}
shape_prop_rules = {
    bool_bailout: lambda b, *, expect: record(None, bool_bailout, b, expect=b.value),
    int_eq: lambda a, b: a == b,
    int_const: lambda *, val: val,
    size_index: lambda s, *, index: s[index],
    size_ctor: lambda *args: args,
    var_placeholder: lambda *, name: env[name],
    var_add: lambda a, b: a,
    var_mul: lambda a, b: a,
    var_size: lambda a: a,
    var_expand: lambda a, size: size,
}
print("---")
sa = GuardedInt(4, "sa")
sb = GuardedInt(4, "sb")
env["a"] = (sa,)
env["b"] = (sb,)
graph = CURRENT_GRAPH
CURRENT_GRAPH = Graph()
interp_graph(graph, shape_prop_rules)


interp_graph(CURRENT_GRAPH, interp_rules, init={"sa": 8, "sb": 1})


"""
def extract_bailout(graph):
    needed = set()
    residual_graph = []
    for n in reversed(graph.nodes):
        if any(o in needed for o in n.outputs) or n.op is bool_bailout:
            residual_graph.append(n)
            for i in n.inputs:
                needed.add(i)
    residual_graph.reverse()
    return Graph(residual_graph)

print(extract_bailout(CURRENT_GRAPH))
"""
