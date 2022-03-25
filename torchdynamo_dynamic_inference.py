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

# +
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

torch.manual_seed(0)
# -

# This notebook explains how Jason Ansel's proposal for very simple
# dynamic shapes in TorchDynamo works in
# https://github.com/facebookresearch/torchdynamo/issues/38
#
# The general model for torchdynamo graphs is that they consist of a
# set of guards plus a trace.  The guards say whether or not the trace
# is valid; if it is not, torchdynamo must redo its analysis and
# recompile the graph in question.
#
# In this simplified model, we will model torchdynamo graphs as a
# dead simple AST (in reality, you need a graph representation to
# specify ordering of operations, sharing and multiple outputs, but
# they are not relevant to this example so I've dumped them.)
#
# First, we define various operations on the graph.  add and mul
# do what you expect: they perform a (broadcasting) PyTorch add and
# mul.  `dynamic_param` and `static_param` both represent inputs
# to the graph.  The distinction is that `dynamic_param` inputs
# correspond to inputs which are fully dynamic: their shapes can
# vary from execution to execution of the graph.  `static_param`
# inputs, on the other hand, are required to be some specific size.
#

# +
@dataclass(frozen=True)
class Op:
    name: str

    def __str__(self):
        return self.name


v_dynamic_param = Op("v_dynamic_param")
v_static_param = Op("v_static_param")
v_add = Op("v_add")
v_mul = Op("v_mul")
# -

# We can stitch these operations together in an AST of expressions
# of operators applied to some other expressions (and possibly some
# other, static metadata).

# +


@dataclass(eq=False)
class Node:
    op: Op
    inputs: List["Node"] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        inputs_str = ", ".join(repr(i) for i in self.inputs)
        params_str = ""
        if self.inputs and self.params:
            params_str += ", "
        params_str += ", ".join(
            f"{k}={v}"
            for k, v in self.params.items()
            if k != "size" and self.op is v_dynamic_param
        )
        return f"{self.op}({inputs_str}{params_str})"


# -

# And then we can write an interpreter for these inputs.  Notice that
# we fetch parameters from an environment that's passed into the
# interpreter; if the parameter is dynamic we pass it in directly,
# but if it's static, we first check that the size of the parameter
# is consistent with the saved size.

# +

INTERP_RULES = {}
INTERP_RULES[v_add] = lambda x, y: x + y
INTERP_RULES[v_mul] = lambda x, y: x * y


def interp_node(n: Node, env: Dict[Node, torch.Tensor]):
    if n.op is v_dynamic_param:
        return env[n.params["name"]]
    elif n.op is v_static_param:
        r = env[n.params["name"]]
        assert (
            r.shape == n.params["size"]
        ), f"static shape mismatch: {r.shape} and {n.params['size']}"
        return r
    args = [interp_node(i, env) for i in n.inputs]
    return INTERP_RULES[n.op](*args, **n.params)


# -

# In actual torchdynamo, we can construct our IR directly via
# bytecode analysis.  But this isn't really necessary for our
# example here; we can use an ordinary tracer to construct the IR as
# well.  Our tracer is very simple.


@dataclass
class Variable:
    tensor: torch.Tensor
    node: Node

    # This will be implemented later
    def size(self):
        return variable_size(self)

    @staticmethod
    def param(tensor: torch.Tensor, name: str):
        # Save the observed shape, but by default dynamic_param won't
        # check it!
        return Variable(
            tensor, Node(v_dynamic_param, [], {"name": name, "size": tensor.shape})
        )

    def __mul__(self, rhs: "Variable") -> "Variable":
        r_tensor = self.tensor * rhs.tensor
        r_node = Node(v_mul, [self.node, rhs.node])
        return Variable(r_tensor, r_node)

    def __add__(self, rhs: "Variable") -> "Variable":
        r_tensor = self.tensor + rhs.tensor
        r_node = Node(v_add, [self.node, rhs.node])
        return Variable(r_tensor, r_node)


# With this, we can run a simple example, print out the IR for it,
# and then rerun it.  By default, we treat the inputs as dynamics,
# so we are allowed to rerun the IR even though the input sizes have
# changed (because there is nothing shape specific in the IR.)

a = Variable.param(torch.randn(4), "a")
b = Variable.param(torch.randn(4), "b")
r = a * b

print(r.node)

print(interp_node(r.node, {"a": torch.randn(5), "b": torch.randn(1)}))

# Now, the problem is what happens if a user wants to vary the
# behavior of their computation based on the size of their input?
# Then our trace is no longer valid in this situation!
#
# torchdynamo deals with this situation by looking for explicit uses
# of sizes.  If there is an explicit use of a size, it goes ahead
# and conservatively marks all of the parameters which could have
# contributed to the size of this tensor as static, indicating that
# the trace is now only valid for those specific sizes.

# +


def input_sources(node):
    r = set()
    for i in node.inputs:
        r |= input_sources(i)
    if node.op is v_dynamic_param:
        r.add(node)
    return r


def variable_size(self):
    for i in input_sources(self.node):
        # change it from dynamic to static.  (the parameter
        # already saved the size, we don't need to recover it)
        i.op = v_static_param
    return self.tensor.size()


# -

# Now if we have dependent control flow on an input, we will
# appropriately fail if you pass in mismatching sizes.

# +

a = Variable.param(torch.randn(4), "a")
b = Variable.param(torch.randn(4), "b")
if a.size()[0] == 4:
    r = a + b
else:
    r = a * b

# -

print(r.node)

print(interp_node(r.node, {"a": torch.randn(4), "b": torch.randn(4)}))

try:
    print(interp_node(r.node, {"a": torch.randn(5), "b": torch.randn(1)}))
except Exception:
    traceback.print_exc()

# It will still work even if the shape check is done an intermediate
# computation (in this case, both a and b are marked as dynamic).

# +

a = Variable.param(torch.randn(1), "a")
b = Variable.param(torch.randn(1), "b")
c = a + b
if c.size()[0] == 1:
    r = a + c
else:
    r = a * c
# -

print(r.node)

try:
    print(interp_node(r.node, {"a": torch.randn(1), "b": torch.randn(5)}))
except Exception:
    traceback.print_exc()

# This analysis is VERY conservative.  Although there are some easy
# improvements you can apply, you are limited in the precision you can
# have without having shape formulas for operators that can propagate
# dynamic shapes.  With shape formulas, you can track exact dependencies
# on a size-by-size basis; if you matrix multiply two tensors C = A @ B,
# a use of C.shape[0] will only add a guard for A.shape[0], and a use of
# C.shape[1] will only add a guard for B.shape[1].  The analysis here
# will just make both A and B static, and we cannot do any better
# without more knowledge of formulas.  This suggests that an important
# workstream to improve precision is to get dynamic-aware shape formulas
# in Python for as many operators as possible.
