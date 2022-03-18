# -*- coding: utf-8 -*-

from typing import List, NamedTuple, Callable, Dict, Optional, Union, Any
import torch
from dataclasses import dataclass, field
import functools

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
# TBH, it is up to you whether or not you actually want to
# natively support booleans; in this pedagogical implementation
# it makes things a little simpler, but you can also fuse the
# asserts with the operation on integers
fresh_bool = FreshSupply('b')

@dataclass(frozen=True)
class Op:
    name: str

bool_assert = Op("bool_assert")
int_eq = Op("int_eq")
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

@register(INTERP_RULES, bool_assert)
def interp_bool_assert(b: bool):
    assert b

INTERP_RULES[int_eq] = lambda x, y: x == y
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

@dataclass
class Node:
    op: Op
    inputs: List[Atom]
    outputs: List[str]
    params: Dict[str, Any] = field(default_factory=dict)

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

# Constraints add nodes to the graph which specify constraints
# on sizes which may not have been evident upon entry

CURRENT_GRAPH = Graph()

def record_arg(a):
    if isinstance(a, tuple):
        return tuple(record_arg(b) for b in a)
    elif isinstance(a, int):
        return a
    else:
        assert isinstance(a, (Variable, SymbolicIntNode, SymbolicBoolNode))
        return a.name

def record_var(op, shape, *args, name=None, **kwargs):
    r = Variable(shape, name=name)
    CURRENT_GRAPH.nodes.append(Node(op, tuple(record_arg(a) for a in args), [r.name], kwargs))
    return r

def record_none(op, *args, **kwargs):
    CURRENT_GRAPH.nodes.append(Node(op, tuple(record_arg(a) for a in args), [], kwargs))

class SymbolicBoolNode:
    name: str
    def __init__(self):
        self.name = fresh_bool()
    def __repr__(self):
        return self.name

SymBool = Union[SymbolicBoolNode, bool]

def record_bool(op, *args, **kwargs):
    b = SymbolicBoolNode()
    CURRENT_GRAPH.nodes.append(Node(op, tuple(record_arg(a) for a in args), [b.name], kwargs))
    return b

class SymbolicIntNode:
    name: str
    def __init__(self):
        self.name = fresh_int()
    def __repr__(self):
        return self.name
    def __eq__(self, other):
        # Don't spuriously attempt equality with other things
        if not isinstance(other, int) and not isinstance(other, SymbolicIntNode):
            return NotImplemented
        # peephole optimization.  this would do even better if we
        # did unification on the go
        if isinstance(other, SymbolicIntNode) and self.name == other.name:
            return True
        return record_bool(int_eq, self, other)

SymInt = Union[SymbolicIntNode, int]

def record_int(op, *args, **kwargs):
    CURRENT_GRAPH.nodes.append(Node(op, tuple(a.name for a in args), [], kwargs))

# if we're cool kids like FX we can use bytecode analysis to interpose
# on asserts, but since we're not we just implement it manually
def assert_(b: SymBool):
    if isinstance(b, bool):
        assert b
    else:
        record_none(bool_assert, b)

def assert_shape_eq(lhs, rhs):
    # We are rank specialized, so this is static
    assert lhs.dim() == rhs.dim(), f"{lhs}: {lhs.shape}, {rhs}: {rhs.shape}"
    for x, y in zip(lhs.shape, rhs.shape):
        # peephole optimization
        assert_(x == y)

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
    assert_shape_eq(self, rhs)
    r = record_var(var_mul, self.shape, self, rhs)
    print(f'{r.name}: {r.shape} = {self.name} * {rhs.name}')

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
    assert_shape_eq(self, rhs)
    r = record_var(var_add, self.shape, self, rhs)
    print(f'{r.name}: {r.shape} = {self.name} + {rhs.name}')
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
    print(f'{r.name}: {r.shape} = {self.name}.sum()')
    def propagate(dL_doutputs: List[Variable]):
        dL_dr, = dL_doutputs
        size = self.shape
        return [dL_dr.expand(size)]
    gradient_tape.append(TapeEntry(inputs=[self.name], outputs=[r.name], propagate=propagate))
    return r


def operator_expand(self: Variable, sizes: List[SymInt]) -> 'Variable':
    assert self.dim() == 0 # only works for scalars
    r = record_var(var_expand, sizes, self, sizes)
    print(f'{r.name}: {r.shape} = {self.name}.expand({sizes})')
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
loss = simple(a, b).sum()
da, db = grad(loss, [a, b])
print("da", da)
print("db", db)

# Now let's do it again but symbolic

print("===========")
reset_tape()
s1 = SymbolicIntNode()
s2 = SymbolicIntNode()
a = Variable((s1,))
b = Variable((s2,))
loss = simple(a, b).sum()
da, db = grad(loss, [a, b])
print("da", da)
print("db", db)
