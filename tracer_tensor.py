import torch
from torch import Tensor
from torch.fx import Tracer, GraphModule, Graph
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import tree_map
from torch.testing._internal.common_utils import TestCase, run_tests

from utils import no_dispatch, tree_map2
from types import FunctionType
from base_tensor import BaseTensor

# This is a reimplementation of Horace He's original
# PythonTensor in functorch:
# https://github.com/pytorch/functorch/blob/main/functorch/_src/python_key.py


class TracerTensor(BaseTensor):
    __slots__ = ['proxy']

    # We support autograd-ing through the TracerTensor (which you
    # really can think of as a good old fashioned tensor that also
    # takes a proxy along for the ride).  If you need to terminate
    # the autograd early, use torch.autograd.grad with explicit
    # inputs.
    @staticmethod
    def __new__(cls, elem, proxy):
        return super().__new__(cls, elem)

    def __init__(self, elem, proxy):
        # elem does not need to be recorded, because TracerTensor *is a* elem
        self.proxy = proxy
        # Since the proxy is associated with a concrete Tensor object, we also
        # know exactly what its tensor metadata should be, so populate it
        proxy.node.meta['tensor_meta'] = _extract_tensor_metadata(self)

    def __repr__(self):
        with no_dispatch():
            return f"TracerTensor({repr(self)})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_proxy(t):
            if isinstance(t, cls):
                return t.proxy
            else:
                return t

        def wrap(t, p):
            # Some ops (like native_batch_norm_backward) return undefined
            # tensor that get converted into None in Python.  This papers
            # over this problem.
            # TODO: fix this inside core so that None returns from
            # __torch_dispatch__ turn into undefined tensors
            if t is None:
                return torch.empty(())
            elif isinstance(t, Tensor):
                return TracerTensor(t, p)
            else:
                assert t == p
                return t

        r = super().__torch_dispatch__(func, types, args, kwargs)

        r_proxy = func(
            *tree_map(unwrap_proxy, args),
            **tree_map(unwrap_proxy, kwargs))

        # NB: we cannot zip r and r_proxy, or rely on r_proxy knowing its
        # structure, because r_proxy as implemented in FX typically is a proxy
        # that will generate IR for accessing subfields as you invoke.  So
        # r has to "drive" the deconstruction.
        # NB: this assumes there aren't any recursive return structs, which
        # is generally a safe bet in the current codegen
        if isinstance(r, list):
            return [wrap(t, r_proxy[i]) for i, t in enumerate(r)]
        elif isinstance(r, tuple):
            return tuple(wrap(t, r_proxy[i]) for i, t in enumerate(r))
        else:
            return wrap(r, r_proxy)


class DispatchTracer(Tracer):
    # This is modeled off of Trace.trace but we don't need to monkeypatch
    # anything because we will rely on __torch_dispatch__ to handle
    # interposition.  Unlike standard FX, we don't want to trace leaf modules,
    # we want to get a graph of entirely torch.ops.aten operations
    #
    # Unlike FX, the semantics for concrete_args is a little different.
    # Typically, if you FX trace a function, you leave concrete_args None
    # (because you want most of the arguments to be symbolic).  When we
    # dispatch trace a function, we want the arguments to be concrete because
    # they are going to advertise as honest to goodness tensors (if you want
    # to avoid actually doing the compute while tracing, you should pass in
    # meta tensors).
    def trace(self, root, concrete_args):
        # TODO: add torch.nn.Module support (??)
        assert not isinstance(root, torch.nn.Module)
        self.root = torch.nn.Module()
        fn = root

        tracer_cls = getattr(self, '__class__', None)
        self.graph = Graph(tracer_cls=tracer_cls)
        # No module, so tensor_attrs is always empty
        self.tensor_attrs = {}
        assert isinstance(fn, FunctionType)

        # Reimplementation of create_args_for_root, but this is pretty
        # different as we need a blend of concrete/non-concrete behavior.
        # PH is irrelevant.
        # NB: Didn't bother handling functools wrappers
        # TODO: add back argument name sniffing
        cnt = 0
        def replace_tracer(arg):
            nonlocal cnt
            cnt += 1
            return TracerTensor(arg, self.create_proxy('placeholder', f'arg_{str(cnt)}', (), {}))
        args = tree_map(replace_tracer, concrete_args)

        self.create_node('output', 'output', (self.create_arg(fn(*args).proxy),), {},
                         type_expr=fn.__annotations__.get('return', None))

        self.submodule_paths = None
        return self.graph

def dispatch_trace(root, concrete_args):
    tracer = DispatchTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__name__
    return GraphModule(tracer.root, graph, name)


class TracerTensorTest(TestCase):
    def test_basic(self):
        g = dispatch_trace(lambda x, y: x + y, (torch.ones(2), torch.ones(2)))
        self.assertExpectedInline(str(g.graph), """\
graph():
    %arg_1 : [#users=1] = placeholder[target=arg_1]
    %arg_2 : [#users=1] = placeholder[target=arg_2]
    %add : [#users=1] = call_function[target=torch.ops.aten.add](args = (%arg_1, %arg_2), kwargs = {})
    return add""")


if __name__ == '__main__':
    run_tests()
