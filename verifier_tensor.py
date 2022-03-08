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
    def __init__(self, node):
        self.node = node

    def advance(self):
        node = self.node
        self.node = node.next
        return node

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
        assert node.op == "call_function"
        # TODO: how come the saved target in FX isn't an overload and is a
        # packet???
        assert node.target == func.overloadpacket

        for i, v in enumerate(node.args):
            if isinstance(v, Node):
                assert isinstance(args[i], VerifierTensor)
                assert v is args[i].node
            else:
                assert v == args[i]

        for k, v in node.kwargs.items():
            if isinstance(v, Node):
                assert isinstance(kwargs[k], VerifierTensor)
                assert v is kwargs[k].node
            else:
                assert v == kwargs[k]
        assert len(node.kwargs) == len(kwargs)

        r = super().__torch_dispatch__(func, types, args, kwargs)

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
            global VERIFIER
            VERIFIER = Verifier(next(iter(self.graph.graph.nodes)))
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
        f(torch.ones(2), torch.zeros(2))

if __name__ == '__main__':
    run_tests()
