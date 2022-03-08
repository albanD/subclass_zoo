import torch
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.fx import Interpreter, Node

from tracer_tensor import dispatch_trace
from base_tensor import BaseTensor

# how to do speculate and validate
#   - need a function under trace (dispatch_trace)
#   - first time run with normal TracerTensor
#   - second time run with VerifierTensor
# recovery is not necessary

class Verifier:
    def __init__(self, interpreter, node):
        self.node = node
        # We aren't actually going to run the interpreter, it's just
        # here for fetch_attr
        self.interpreter = interpreter
        # TODO: IDK if there's a better way to do this
        self.constant_map = {}

    def advance(self):
        node = self.node
        self.node = node.next

        while node.op == "get_attr":
            self.constant_map[self.interpreter.fetch_attr(node.target)] = node
            node = self.node
            self.node = node.next

        return node

    def constant_node(self, t):
        return self.constant_map[t]

VERIFIER = None

class VerifierTensor(BaseTensor):
    @staticmethod
    def __new__(cls, elem, node):
        return super().__new__(cls, elem)

    def __init__(self, elem, node):
        self.node = node

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Verify that this is correct
        node = VERIFIER.advance()
        assert node.op == "call_function", node.op
        # TODO: this is a bug, the saved function in FX should be an overload,
        # not an overload packet
        assert node.target == func.overloadpacket

        def translate(n, v):
            if isinstance(n, Node):
                if isinstance(v, VerifierTensor):
                    assert n is v.node
                    return v
                else:
                    assert n is VERIFIER.constant_node(v)
                    return v.to("meta")
            else:
                assert n == v
                return v

        meta_args = []
        meta_kwargs = {}
        for i, n in enumerate(node.args):
            meta_args.append(translate(n, args[i]))
        for k, n in node.kwargs.items():
            meta_kwargs[k] = translate(n, kwargs[k])
        assert len(node.kwargs) == len(kwargs)

        r = super().__torch_dispatch__(func, types, tuple(meta_args), meta_kwargs)

        # For the multi-outputs need to advance verifier past the indexing
        # nodes
        if isinstance(r, list):
            raise NotImplementedError
        elif isinstance(r, tuple):
            raise NotImplementedError
        else:
            return VerifierTensor(r, node)


class SpeculatingJit:
    def __init__(self, root):
        self.root = root
        self.graph = None
        self.interpreter = None

    def transform(self, graph):
        return graph

    def __call__(self, *args):
        if self.graph is None:
            r, self.graph = dispatch_trace(self.root, args)
            self.interpreter = Interpreter(self.transform(self.graph))
            return r
        else:
            # assume the placeholder nodes are first
            # TODO: there is a problem with the verifier design here which
            # is that it is not possible to free constants that are captured
            # by the graph, which might be important for memory usage
            # if FX transformation did weight transformation.  I think what
            # you want to do is stub out the tensors with meta "shadows"
            # that have a correspondence to getattr nodes but it is a little
            # fiddly to implement
            global VERIFIER
            VERIFIER = Verifier(Interpreter(self.graph), next(iter(self.graph.graph.nodes)))
            i = 0
            verifier_args = []
            for a in args:
                n = VERIFIER.advance()
                assert n.op == "placeholder"
                verifier_args.append(VerifierTensor(a.to("meta"), n))
            r = self.interpreter.run(*args)
            verifier_r = self.root(*verifier_args)
            VERIFIER = None
            assert r.shape == verifier_r.shape
            assert r.dtype == verifier_r.dtype
            return r


class VerifierTensorTest(TestCase):
    def test_basic(self):
        def root(x, y):
            # TODO: x + y is annoying to debug because the exception gets
            # swallowed
            return torch.add(x, y)

        f = SpeculatingJit(root)
        r = f(torch.zeros(2), torch.zeros(2))
        self.assertEqual(r, torch.zeros(2))
        r2 = f(torch.ones(2), torch.zeros(2))
        self.assertEqual(r2, torch.ones(2))

    def test_constant(self):
        x = torch.zeros(2)
        def root(y):
            return torch.add(x, y)

        f = SpeculatingJit(root)
        r = f(torch.zeros(2))
        self.assertEqual(r, torch.zeros(2))
        r2 = f(torch.ones(2))
        self.assertEqual(r2, torch.ones(2))

    def test_validation_failure(self):
        i = 0
        def root(x, y):
            nonlocal i
            i += 1
            if i == 1:
                return torch.add(x, y)
            else:
                return torch.mul(x, y)

        f = SpeculatingJit(root)
        r = f(torch.zeros(2), torch.zeros(2))
        self.assertEqual(r, torch.zeros(2))
        self.assertRaises(AssertionError, lambda: f(torch.ones(2), torch.zeros(2)))

if __name__ == '__main__':
    run_tests()
