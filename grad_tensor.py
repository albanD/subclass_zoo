import torch
from torch import Tensor
from typing import List, NamedTuple, Callable, Dict, Optional

class TapeEntry(NamedTuple):
    # names of the inputs to the original computation
    inputs : List[str]
    # names of the outputs of the original computation
    outputs: List[str]
    # apply chain rule
    propagate: Callable[List[Tensor], List[Tensor]]

class Torch:
    def mul(self, lhs, rhs):
        return torch.mul(lhs, rhs)
    def add(self, lhs, rhs):
        return torch.add(lhs, rhs)
    def sum(self, input):
        return torch.sum(input)
    def expand(self, input, sizes):
        return input.expand(*sizes)

def gen_autograd(suffix="", *, backward_super: bool = False):
    class Autograd:
        def get__name(self):
            return getattr(self, f"_name_{Autograd.__name__}")
        def set__name(self, x):
            return setattr(self, f"_name_{Autograd.__name__}", x)
        _name = property(get__name, set__name)

        def get_gradient_tape(self):
            return getattr(self, f"gradient_tape_{Autograd.__name__}")
        def set_gradient_tape(self, x):
            return setattr(self, f"gradient_tape_{Autograd.__name__}", x)
        gradient_tape = property(get_gradient_tape, set_gradient_tape)

        def __init__(self):
            self._name = 0
            self.gradient_tape = []

        def fresh_name(self) -> str:
            """ create a new unique name for a variable: v0, v1, v2 """
            r = f'v{self._name}'
            self._name += 1
            return r

        def variable(self, t: Tensor, name: str=None):
            #t = t.detach()
            if not hasattr(t, "t_name"):
                t.t_name = name or self.fresh_name()
            return t

        def constant(self, value: torch.Tensor, name: str=None):
            r = self.variable(value, name)
            print(f'{Autograd.__name__} {r.t_name} = {value}')
            return r

        def backward_dispatch(self, cb):
            if backward_super:
                return self.variable(cb(super()))
            else:
                return cb(self)

        def mul(self, lhs, rhs):
            if isinstance(rhs, float) and rhs == 1.0:
                # peephole optimization
                return lhs

            # define forward
            r = self.variable(super().mul(lhs, rhs))
            print(f'{Autograd.__name__} {r.t_name} = {lhs.t_name} * {rhs.t_name}')

            # record what the inputs and outputs of the op were
            inputs = [lhs.t_name, rhs.t_name]
            outputs = [r.t_name]

            # define backprop
            def propagate(dL_doutputs: List[Tensor]):
                dL_dr, = dL_doutputs

                dr_dlhs = rhs # partial derivative of r = lhs*rhs
                dr_drhs = lhs # partial derivative of r = lhs*rhs

                # chain rule propagation from outputs to inputs of multiply
                # self or Autograd???
                dL_dlhs = self.backward_dispatch(lambda s: s.mul(dL_dr, dr_dlhs))
                dL_drhs = self.backward_dispatch(lambda s: s.mul(dL_dr, dr_drhs))
                dL_dinputs = [dL_dlhs, dL_drhs]
                return dL_dinputs
            # finally, we record the compute we did on the tape
            self.gradient_tape.append(TapeEntry(inputs=inputs, outputs=outputs, propagate=propagate))
            return r

        def add(self, lhs, rhs):
            # Add follows a similar pattern to Mul, but it doesn't end up
            # capturing any variables.
            r = self.variable(super().add(lhs, rhs))
            print(f'{Autograd.__name__} {r.t_name} = {lhs.t_name} + {rhs.t_name}')
            def propagate(dL_doutputs: List[Tensor]):
                dL_dr, = dL_doutputs
                dr_dlhs = 1.0
                dr_drhs = 1.0
                dL_dlhs = self.backward_dispatch(lambda s: s.mul(dL_dr, dr_dlhs))
                dL_drhs = self.backward_dispatch(lambda s: s.mul(dL_dr, dr_drhs))
                return [dL_dlhs, dL_drhs]
            self.gradient_tape.append(TapeEntry(inputs=[lhs.t_name, rhs.t_name], outputs=[r.t_name], propagate=propagate))
            return r

        def sum(self, input: Tensor, name: Optional[str]=None):
            r = self.variable(super().sum(input), name=name)
            print(f'{Autograd.__name__} {r.t_name} = {input.t_name}.sum()')
            def propagate(dL_doutputs: List[Tensor]):
                dL_dr, = dL_doutputs
                size = input.size()
                return [self.backward_dispatch(lambda s: s.expand(dL_dr, size))]
            self.gradient_tape.append(TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate))
            return r

        def expand(self, input: Tensor, sizes: List[int]):
            assert(input.dim() == 0) # only works for scalars
            r = self.variable(super().expand(input, sizes))
            print(f'{Autograd.__name__} {r.t_name} = {input.t_name}.expand({sizes})')
            def propagate(dL_doutputs: List[Tensor]):
                dL_dr, = dL_doutputs
                return [self.backward_dispatch(lambda s: s.sum(dL_dr))]
            self.gradient_tape.append(TapeEntry(inputs=[input.t_name], outputs=[r.t_name], propagate=propagate))
            return r

        def reset_tape(self):
            self.gradient_tape.clear()
            self._name = 0 # reset variable names too to keep them small.

        def grad(self, L, desired_results: List[Tensor]) -> List[Tensor]:
            # this map holds dL/dX for all values X
            dL_d : Dict[str, Tensor] = {}
            # It starts by initializing the 'seed' dL/dL, which is 1
            dL_d[L.t_name] = self.variable(torch.ones(()))
            print(f'{Autograd.__name__} d{L.t_name} ------------------------')

            # look up dL_dentries. If a variable is never used to compute the loss,
            # we consider its gradient None, see the note below about zeros for more information.
            def gather_grad(entries: List[str]):
                return [dL_d[entry] if entry in dL_d else None for entry in entries]

            # propagate the gradient information backward
            for entry in reversed(self.gradient_tape):
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
                        dL_d[input] = self.backward_dispatch(lambda s: s.add(dL_d[input], dL_dinput))

            # print some information to understand the values of each intermediate 
            for name, value in dL_d.items():
                print(f'{Autograd.__name__} d{L.t_name}_d{name} = {value.t_name}')
            print(f'------------------------')

            return gather_grad(desired.t_name for desired in desired_results)
    Autograd.__name__ = f"Autograd{suffix}"
    return Autograd

Autograd = gen_autograd()

# sum is used to turn our matrices into a single scalar to get a loss.
# expand is the backward of sum, so it is added to make sure our Variable
# is closed under differentiation. Both have rules similar to mul above.

torch.manual_seed(0)
a_global, b_global = torch.rand(4), torch.rand(4)

class Example1(Autograd, Torch):
    def simple(self, a, b):
        t = self.add(a, b)
        return self.mul(t, b)

    def main(self):
        a = self.variable(a_global, name='a')
        b = self.variable(b_global, name='b')
        loss = self.simple(a, b)
        da, db = self.grad(loss, [a, b])
        print("da", da)
        print("db", db)

#Example1().main()

class Example2Direct(Autograd, Torch):
    def simple(self, a, b):
        t = self.add(a, b)
        return self.mul(t, b)

    def run_gradients(self):
        a = self.variable(a_global, name='a')
        b = self.variable(b_global, name='b')

        # our first loss
        L0 = self.sum(self.simple(a, b), name='L0')

        # compute derivatives of our inputs
        dL0_da, dL0_db = self.grad(L0, [a, b])

        # now lets compute the L2 norm of our derivatives
        L1 = self.sum(self.add(self.mul(dL0_da, dL0_da), self.mul(dL0_db, dL0_db)), name='L1')

        # and take the gradient of that.
        # notice there are two losses involved.
        dL1_da, dL1_db = self.grad(L1, [a, b])
        return dL1_da, dL1_db

    def main(self):
        da, db = self.run_gradients()
        print("da", da)
        print("db", db)

#Example2Direct().main()

Autograd1 = gen_autograd("1", backward_super=True)
Autograd2 = gen_autograd("2", backward_super=True)
class Example2Indirect(Autograd2, Autograd1, Torch):
    def simple(self, cls, a, b):
        t = cls.add(self, a, b)
        return cls.mul(self, t, b)

    def run_gradients(self):

        # Imagine grad(grad(...))
        #   we first allocate variables for the outer grad (Autograd1)
        #   then they get wrapped in variables again for inner grad (Autograd2)

        a1 = Autograd1.variable(self, a_global, name='a1')
        b1 = Autograd1.variable(self, b_global, name='b1')

        a2 = Autograd2.variable(self, a1, name='a2')
        b2 = Autograd2.variable(self, b1, name='b2')

        L0 = Autograd2.sum(self, self.simple(Autograd2, a2, b2), name='L0')
        dL0_da, dL0_db = Autograd2.grad(self, L0, [a2, b2])

        # Now we can "throw out" the tape for Autograd2
        Autograd2.reset_tape(self)

        # now lets compute the L2 norm of our derivatives, in Autograd1
        L1 = Autograd1.sum(self, Autograd1.add(self, Autograd1.mul(self, dL0_da, dL0_da), Autograd1.mul(self, dL0_db, dL0_db)), name='L1')

        # and take the gradient of that.
        # notice there are two losses involved.
        dL1_da, dL1_db = Autograd1.grad(self, L1, [a1, b1])
        return dL1_da, dL1_db

    def main(self):
        da, db = self.run_gradients()
        print("da", da)
        print("db", db)

Example2Indirect().main()
