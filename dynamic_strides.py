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

import sympy as sp
from typing import List
from dataclasses import dataclass

# In this notebook, we explore how to model contiguity and strides in a
# universe where we support dynamic shapes.  We don't care about dynamic
# strides/contiguity per se (we'd be OK with specializing on the input
# being contiguous, channels-last, etc), but strides and contiguity
# are *derived* from shapes, so if you have dynamic shapes, you
# also end up with dynamic strides and contiguity.
#
# Let's take a concrete look at this phenomenon in the simplest possible
# context: a contiguous tensor.  Here is the C++ code which implements
# computation of contiguous strides for a tensor:

"""
// From c10/util/strides.h
// Computes the contiguous strides of a tensor, given its sizes.
static inline std::vector<typename IntArrayRef::value_type> contiguous_strides(
    const IntArrayRef sizes) {
  using Int = IntArrayRef::value_type;
  const Int dims = static_cast<Int>(sizes.size());

  std::vector<Int> strides;

  if (dims > 0) {
    strides.assign(dims, 0);
    // Start by populating the last dimension: its strides is always 1.
    strides[dims - 1] = 1;
    for (auto i = dims - 2; i >= 0; --i) {
      // Strides can't be 0 even if sizes are 0.
      strides[i] = strides[i + 1] * std::max(sizes[i + 1], Int{1});
    }
  }

  return strides;
}
"""

# And a port to Python:


def contiguous_strides(sizes: List[int]):
    dims = len(sizes)
    strides = []
    if dims > 0:
        strides = [0] * dims
        strides[dims - 1] = 1
        for i in range(dims - 2, -1, -1):
            strides[i] = strides[i + 1] * sp.Max(sizes[i + 1], 1)
    return strides


print(contiguous_strides([2, 3, 5]))

# Let's look at the symbolic output of this function.  When only the batch
# dimension is dynamic, things are pretty simple:

x = sp.symbols("x")
print(contiguous_strides([x, 3, 5]))

# However, if an inner dimension is dynamic, the dynamic shape variable
# shows up in the stride calculation

print(contiguous_strides([2, x, 5]))

# The set of strides returned by contiguous is guaranteed to be
# contiguous, but the inverse is not true: there are some degrees of
# freedom in the definition of strides when sizes are one or zero.
# Here is our definition of "when something is contiguous" (not accounting
# for overflow):

"""
// In c10/core/TensorImpl.h
inline bool is_empty() const {
  return numel() == 0;
}

// In c10/core/TensorImpl.cpp
bool TensorImpl::compute_contiguous() const {
  bool is_contiguous = true;
  if (is_empty())
    return is_contiguous;
  int64_t z = 1;
  for (int64_t d = dim() - 1; d >= 0; d--) {
    const auto size_d = sizes_and_strides_.size_at_unchecked(d);
    if (size_d != 1) {
      if (sizes_and_strides_.stride_at_unchecked(d) == z) {
        z *= size_d;
      } else {
        is_contiguous = false;
        break;
      }
    }
  }
  return is_contiguous;
}
"""

# In Python (note that we will use the suffix branchy to refer
# to code which branches on the concrete value of sizes/strides):


def compute_numel(sizes: List[int]):
    numel = 1
    for s in sizes:
        numel *= s
    return numel


def compute_contiguous_branchy(sizes: List[int], strides: List[int]):
    is_contiguous = True
    if compute_numel(sizes) == 0:
        return is_contiguous
    z = 1
    for d in range(len(sizes) - 1, -1, -1):
        if sizes[d] != 1:
            if strides[d] == z:
                z *= sizes[d]
            else:
                is_contiguous = False
                break
    return is_contiguous


# When a dimension has size 1, we are indifferent to the stride at that
# dimension:

print(contiguous_strides([3, 1, 5]))

print(compute_contiguous_branchy([3, 1, 5], [5, 5, 1]))
print(compute_contiguous_branchy([3, 1, 5], [5, 999999, 1]))

# When a tensor contains zero elements, we are indifferent to all the
# strides

print(contiguous_strides([3, 0, 5]))

print(compute_contiguous_branchy([3, 0, 5], [5, 5, 1]))
print(compute_contiguous_branchy([3, 0, 5], [123456, 999999, 424242]))

# Can we compute_contiguous symbolically?  Unfortunately, the "branchy"
# implementation, as written above cannot be run directly on SymPy
# integers, as in several points in the code we condition on the
# concrete values of various comparisons on integers.  Fortunately,
# we can introduce a SymInt/SymBool abstraction (as done in previous
# notebooks) to provide concrete values and record guards expressing
# what is required to be true for the computation to be correct.

# +

GUARDS = []


def is_constant(e):
    if hasattr(e, "is_constant"):
        return e.is_constant()
    elif e is sp.true or e is sp.false:
        return True
    else:
        return False


class SymObject:
    def __post_init__(self):
        if self.expr is None:
            self.expr = sp.sympify(self.val)
        elif not isinstance(self.expr, sp.Expr):
            self.expr = sp.sympify(self.expr)


@dataclass
class SymBool(SymObject):
    val: bool
    expr: sp.Expr = None
    guarded: bool = False

    def __bool__(self):
        if not self.guarded:
            self.guarded = True
            if not is_constant(self.expr):
                if self.val:
                    GUARDS.append(self.expr)
                else:
                    GUARDS.append(sp.Not(self.expr))
        return self.val


def logical_and(self: bool, other: bool):
    if isinstance(self, SymBool) and isinstance(other, SymBool):
        return SymBool(self.val and other.val, sp.And(self.expr, other.expr))
    return sp.And(self, other)


def logical_or(self: bool, other: bool):
    if isinstance(self, SymBool) and isinstance(other, SymBool):
        return SymBool(self.val or other.val, sp.Or(self.expr, other.expr))
    return sp.Or(self, other)


@dataclass
class SymInt(SymObject):
    val: int
    expr: sp.Expr = None
    guarded: bool = False

    def __int__(self):
        if not self.guarded:
            self.guarded = True
            if not is_constant(self.expr):
                GUARDS.append(self.Eq(self.expr, self.val).simplify())
        return self.val

    def __eq__(self, other):
        if not isinstance(other, SymInt):
            other = SymInt(other)
        return SymBool(self.val == other.val, sp.Eq(self.expr, other.expr))

    def __ne__(self, other):
        if not isinstance(other, SymInt):
            other = SymInt(other)
        return SymBool(self.val != other.val, sp.Ne(self.expr, other.expr))

    def __mul__(self, other):
        if not isinstance(other, SymInt):
            other = SymInt(other)
        return SymInt(self.val * other.val, sp.Mul(self.expr, other.expr))

    def __rmul__(self, other):
        if not isinstance(other, SymInt):
            other = SymInt(other)
        return SymInt(self.val * other.val, sp.Mul(self.expr, other.expr))


def I(val, expr=None):
    return SymInt(val, expr)


# -

# Let's run our example.  Under the guards model, we must provide
# concrete values for every symbolic integer, so we can resolve
# conditionals.

x1, x2, x3, y1, y2, y3 = sp.symbols("x1 x2 x3 y1 y2 y3")

GUARDS.clear()
print(
    compute_contiguous_branchy(
        [I(3, x1), I(1, x2), I(5, x3)], [I(5, y1), I(99999, y2), I(1, y3)]
    )
)

# We see that this tensor is contiguous...

print(GUARDS)

# ...subject to these conditions.  These conditions say which particular
# path through the loop we took: we require the sizes to be nonzero,
# there are number of size one equalities/disequalities, and the
# equality requirement between y1 and x3 is the "true" contiguity
# requirement.

# If we are willing to rewrite the definition of compute contiguous, we
# can eliminate the branches, giving a symbolic expression with no
# guards.


def compute_contiguous(sizes, strides):
    is_contiguous = True
    z = 1
    for d in range(len(sizes) - 1, -1, -1):
        is_contiguous = logical_and(
            is_contiguous, logical_or(sp.Eq(sizes[d], 1), sp.Eq(strides[d], z))
        )
        z *= sizes[d]
    return logical_or(sp.Eq(compute_numel(sizes), 0), is_contiguous)


# TODO: prove these two implementations are equivalent, somehow

# We can see that no matter the choice of the stride for a size one
# dimension, the result is always contiguous:

print(compute_contiguous([3, 1, 5], [5, x, 1]))

# And we can see the unflattened contiguity requirement for a completely
# general size/stride tensor.

print(compute_contiguous([x1, x2, x3], [y1, y2, y3]))

# There's other stuff too:
#
#   - We are not "just" compute_contiguous; we also have have variations
#     of this for every memory layout we support.  So the same exercise
#     needs to apply everywhere.
#
#   - We also have non_overlapping_and_dense which which involves a sort
#     which is very annoying.

# In conclusion:
#
#   - We have an explicit choice whether or not to branch inside
#     implementations of code that may be traced.  More trace friendly
#     code is not as good for eager execution (because you can't do
#     things like short circuit).
#
#   - If we store SymInt inside TensorImpl, we need to make a call about
#     how we represent the contiguity bits inside Tensor.  These bits
#     are literally a single bit, so we cannot store a symbolic boolean
#     in them.  It seems the easiest fix is to ensure the
#     is_contiguous() is virtualized (it is), and then internally run
#     (and cache) the symbolic formula done here.
