import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import PythonKeyTracer, ProxyTorchDispatchMode
from torch.fx import Graph, GraphModule

# Limitations:
#   - initialization cannot refer to external tensors
#   - parameters are these weird ProxyTensors, should have a custom class for
#     these placeholders
#   - DCE is likely not sound, needs to be implemented more carefully by
#     understanding aliasing relationships
#   - only top level module is rematerialized
#   - we lose parameter-ness and requires_grad-ness
#   - no version counter safety to guard against input mutation

def deferred_init(f, *args, **kwargs):
    fx_tracer = PythonKeyTracer()
    fx_tracer.graph = Graph(fx_tracer)
    fx_tracer.root = torch.nn.Module()
    fx_tracer.tensor_attrs = {}
    fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
    proxy_mode = ProxyTorchDispatchMode(fx_tracer, trace_factory_functions=True)
    with fake_tensor_mode, proxy_mode:
        r = f(*args, **kwargs)
        r._deferred = fx_tracer
        return r

def materialize_module(m):
    # TODO: handle children

    outputs = []

    def mark_for_materialize(tensors):
        for k, t in tensors.items():
            if t is None:
                continue
            outputs.append(t.proxy.node)

    mark_for_materialize(m._parameters)
    mark_for_materialize(m._buffers)

    m._deferred.graph.output(outputs)
    m._deferred.graph.eliminate_dead_code()  # hmmm
    recomp = GraphModule(m._deferred.root, m._deferred.graph)
    results = recomp()
    results_iter = iter(results)

    def replace_results(tensors):
        for k, t in tensors.items():
            if t is None:
                continue
            tensors[k] = next(results_iter)

    replace_results(m._parameters)
    replace_results(m._buffers)

    del m._deferred


m = deferred_init(torch.nn.Linear, 3, 5)
print(m.weight)
materialize_module(m)
print(m.weight)
