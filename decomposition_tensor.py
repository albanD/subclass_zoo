
from functools import wraps
import torch
from torch._decomp import decomposition_table
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor, capture_logs
from torch.testing._internal.common_utils import run_tests, TestCase

# Goals:
# - we want something reusable that can compose with any subclass
# Non-goal:
# - should work with both __torch_dispatch__ and __torch_function__
#   - decomposition table is aten to aten (alternatively, parametrize on decomposition table?)
#
# Should this be a wrapper subclass ("has-a") or just use inheritance? Both are implemented below.
# - Wrapper subclass API `DecompositionTensor(LoggingTensor(t))`
#   - will need to redispatch
# - OR provide helper dynamically inherit from the provided subclass and just override its __torch_dispatch__
#   function to the decorated version
#   + won't need to redispatch
# - OR User just writes inline if necessary
#   + more customizability
#
# How does this work with torch function?
# - It doesn't. torch function will try to rewrap the output again and error if we don't disable it
#
# What is progressive lowering tensor and how does this compare?
# - TODO

aten = torch.ops.aten

skip_list = [aten.add.Tensor, aten.to.dtype, aten.div.Tensor, aten.clamp.default]

# 1) Wrapper subclass inline version
class DecompositionTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, e):
        r = torch.Tensor._make_wrapper_subclass(cls, e.shape, dtype=e.dtype, requires_grad=False)
        r.elem = e
        return r

    # We may be able to remove this line in the future when Ed's PR lands
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        def unwrap(e):
            return e.elem if isinstance(e, DecompositionTensor) else e

        def wrap(e):
            return DecompositionTensor(e) if isinstance(e, torch.Tensor) else e

        if func in skip_list:
            # Check skip-list first, so "backend" has a chance to see non-decomposed ops
            return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        elif func in decomposition_table:
            return decomposition_table[func](*args, **kwargs)
        else:
            raise NotImplementedError(f"{func.__name__} does not have a decomposition and is not in skip_list")

# 2) Decorator (Alban's suggestion)
def decompose(skip_list, missing_ops=None):
    def _decompose(f):
        @wraps(f)
        def wrapper(cls, func, types, args=(), kwargs=None):
            if func in skip_list:
                # Functions that the layers below are able to handle
                return f(cls, func, types, args, kwargs)
            elif func in decomposition_table:
                return decomposition_table[func](*args, **kwargs)
            else:
                if missing_ops is not None:
                    missing_ops.add(func.__name__)
                    return f(cls, func, types, args, kwargs)
                else:
                    raise NotImplementedError(f"{func.__name__} does not have a decomposition and is not in skip_list")
        return wrapper
    return _decompose

# 2.1) Using the decorator
class DecompositionTensor2(torch.Tensor):
    @staticmethod
    def __new__(cls, e):
        r = torch.Tensor._make_wrapper_subclass(cls, e.shape, dtype=e.dtype, requires_grad=False)
        r.elem = e
        return r

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    @decompose(skip_list, missing_ops=None)
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore
        def unwrap(e):
            return e.elem if isinstance(e, DecompositionTensor2) else e

        def wrap(e):
            return DecompositionTensor2(e) if isinstance(e, torch.Tensor) else e

        return tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))

# 3) Version using inheritance
def apply_decomposition_before_cls(cls, skip_list, missing_ops=None):
    # skip_list here could be the list of ops that your subclass/transform/backend supports
    # Inherits from cls and then wraps its __torch_dispatch__
    cls_new = type(f'Decomposed{cls.__name__}', (cls,), {})
    # Is this always safe to do? What properties does cls need to have for this to be OK?
    # - cls should not be a plain tenosr, which would not have __torch_dispatch__
    #   or we could just check has_attr (?)
    assert cls is not torch.Tensor
    cls_new.__torch_dispatch__ = classmethod(decompose(skip_list, missing_ops)(cls.__torch_dispatch__.__func__))
    return cls_new


class TestDecompositionTensor(TestCase):
    def test_decompose_logging_tensor(self):
        def f(t):
            return F.hardsigmoid(t.add(t))

        # Start off only with LoggingTensor
        with capture_logs() as logs:
            f(LoggingTensor(torch.tensor(1.)))
        self.assertExpectedInline('\n'.join(logs), """\
$1 = torch._ops.aten.add.Tensor($0, $0)
$2 = torch._ops.aten.hardsigmoid.default($1)""")

        # Now we try with LoggingTensor wrapped with a DecompositionTensor (inline version)
        with capture_logs() as logs1:
            f(DecompositionTensor(LoggingTensor(torch.tensor(1.))))
        # We shouldn't see hardsigmoid here anymore because it has been decomposed!
        self.assertExpectedInline('\n'.join(logs1), """\
$1 = torch._ops.aten.add.Tensor($0, $0)
$2 = torch._ops.aten.to.dtype($1, torch.float32)
$3 = torch._ops.aten.add.Tensor($2, 3)
$4 = torch._ops.aten.clamp.default($3, 0)
$5 = torch._ops.aten.clamp.default($4, None, 6)
$6 = torch._ops.aten.div.Tensor($5, 6)
$7 = torch._ops.aten.to.dtype($6, torch.float32)""")
        # With the decorator version
        with capture_logs() as logs2:
            f(DecompositionTensor2(LoggingTensor(torch.tensor(1.))))

        # Patch an existing class
        DecomposedLoggingTensor = apply_decomposition_before_cls(LoggingTensor, skip_list)

        with capture_logs() as logs3:
            f(DecomposedLoggingTensor(torch.tensor(1.)))

        # How would one obtain the skip_list in the first place without having to iterate
        # through errors and add them one-by-one?
        # - We allow users to pass in an empty set to the decorator/wrapper, and this set would
        #   get populated as the program runs
        missing_ops = set()
        DecomposedLoggingTensor2 = apply_decomposition_before_cls(LoggingTensor, [], missing_ops=missing_ops)
        t = DecomposedLoggingTensor2(torch.tensor(1.))
        with capture_logs() as logs4:
            f(t)
        self.assertEqual(missing_ops, set(str(op).split('aten.')[1] for op in skip_list))
        self.assertTrue(logs1 == logs2 == logs3 == logs4)

if __name__ == "__main__":
    run_tests()
