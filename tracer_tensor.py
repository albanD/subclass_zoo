from types import FunctionType

import torch
from base_tensor import BaseTensor
from torch import Tensor
from torch.fx import Graph, GraphModule, Tracer
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_map

from utils import no_dispatch, tree_map2

"""
TracerTensor is a tensor that traces ATen operations that are performed on it
and writes the resulting trace to FX IR. We extracted this tracing
implementation from Horace He's implementation for AOTAutograd
(https://github.com/pytorch/functorch/blob/main/functorch/_src/python_key.py)
to make it easier for you to see how it is put together. The basic
implementation concept is simple: we run all tensor operations as normal, but
on the side, we also duplicate the operations on FX Proxy objects, which are
then responsible for writing in the results into FX IR. The top level tracing
function dispatch_trace is a modified version of FX's `symbolic_trace`
function: we always take a tuple of concrete Tensor inputs, and we generate
placeholder proxies for all of them and attach them to TracerTensors which we
actually feed into the model.

Tracing with __torch_dispatch__ has some properties which are worth being
aware of:

  - It is able to trace through autograd and other PyTorch subsystems (as they
    are all desugared into lower level calls by the time you get to
    `__torch_dispatch__`. Composite operations (CompositeImplicitAutograd)
    will be desugared by the time you get to trace.
  - It produces FX IR with `torch.ops.aten` nodes (e.g., you will get
    `torch.ops.aten.add.Tensor`, not merely `torch.add`)
  - Unlike FX, it is not able to trace non-Tensor symbolic values (e.g.,
    sizes); these are all specialized to particular ints by the time
    `__torch_dispatch__` is called. Nick Korovaiko is working on removing this
    limitation.
  - In fact, you can think of it as a pure Python implementation of
    torch.jit.trace, except that it outputs FX IR rather than TorchScript IR. 
"""


class TracerTensor(BaseTensor):
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
        proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(self)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap_proxy(t):
            if isinstance(t, cls):
                return t.proxy
            else:
                return t

        def wrap(t, p):
            if isinstance(t, Tensor) and not isinstance(t, cls):
                return cls(t, p)
            else:
                assert t == p
                return t

        # Run the original computation
        r = super().__torch_dispatch__(func, types, args, kwargs)

        # Run the computation on FX proxies to record it into graph
        r_proxy = func(*tree_map(unwrap_proxy, args), **tree_map(unwrap_proxy, kwargs))

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
    # Our implementation here divergences a bit from Horace's.  This version
    # modeled off of Trace.trace but we don't need to monkeypatch anything
    # because we will rely on __torch_dispatch__ to handle interposition.
    # Unlike standard FX, we don't want to trace leaf modules, we want to get
    # a graph of entirely torch.ops.aten operations
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

        tracer_cls = getattr(self, "__class__", None)
        self.graph = Graph(tracer_cls=tracer_cls)
        # Don't support module, so tensor_attrs is always empty
        self.tensor_attrs = {}
        assert isinstance(fn, FunctionType)

        # Reimplementation of create_args_for_root, but this is pretty
        # different as we always expect concrete arguments to be provided
        # and we still generate placeholders for each of them.
        cnt = 0

        def replace_tracer(arg):
            nonlocal cnt
            cnt += 1
            # TODO: add back argument name sniffing
            return TracerTensor(
                arg, self.create_proxy("placeholder", f"arg_{str(cnt)}", (), {})
            )

        # TODO: generalize to tree_map (but this will make verifier_tensor
        # harder to implement)
        args = [replace_tracer(a) for a in concrete_args]

        result = fn(*args)

        self.create_node(
            "output",
            "output",
            (self.create_arg(result.proxy),),
            {},
            type_expr=fn.__annotations__.get("return", None),
        )

        self.submodule_paths = None

        # Unlike regular Tracer.trace, we also return the result as it
        # contains useful data (the result of your computation)
        # TODO: better idiom for this
        with no_dispatch():
            unwrapped_result = result.view(result.shape)
        return unwrapped_result, self.graph


def dispatch_trace(root, concrete_args):
    tracer = DispatchTracer()
    result, graph = tracer.trace(root, concrete_args)
    name = root.__name__
    return result, GraphModule(tracer.root, graph, name)


class TracerTensorTest(TestCase):
    def test_basic(self):
        r, g = dispatch_trace(lambda x, y: x + y, (torch.ones(2), torch.ones(2)))
        self.assertNotIsInstance(r, TracerTensor)
        self.assertEqual(r, torch.tensor([2.0, 2.0]))
        self.assertExpectedInline(
            str(g.graph),
            """\
graph():
    %arg_1 : [#users=1] = placeholder[target=arg_1]
    %arg_2 : [#users=1] = placeholder[target=arg_2]
    %add : [#users=1] = call_function[target=torch.ops.aten.add](args = (%arg_1, %arg_2), kwargs = {})
    return add""",
        )

    def test_constant(self):
        x = torch.ones(2)
        _, g = dispatch_trace(lambda y: x + y, (torch.ones(2),))
        self.assertExpectedInline(
            str(g.graph),
            """\
graph():
    %arg_1 : [#users=1] = placeholder[target=arg_1]
    %_tensor_constant0 : [#users=1] = get_attr[target=_tensor_constant0]
    %add : [#users=1] = call_function[target=torch.ops.aten.add](args = (%_tensor_constant0, %arg_1), kwargs = {})
    return add""",
        )


if __name__ == "__main__":
    run_tests()
