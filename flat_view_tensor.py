import torch
from torch.utils._pytree import tree_map
from torch import nn
from torch.nn import Parameter
from typing import List

# TODO: support tensor methods
class IndirectTensor:
    def __init__(self, indir):
        self.indir = indir

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # TODO: you could imagine some sort of caching mechanism, but
        # then you'd also need to design an invalidation mechanism
        def resolve_indir(t):
            if isinstance(t, IndirectTensor):
                return t.indir()
            else:
                return t

        return func(*tree_map(resolve_indir, args), **tree_map(resolve_indir, kwargs))

    # TODO: need to handle grad
    # TODO: need to handle serialization


# model:
#   - as far as autograd is concerned, flat parameter is the only leaf
#   - as far as optimizer is concerned, real parameters are the only
#     parameters


class FlattenParamsWrapper(nn.Module):
    def __init__(self, module, param_buckets: List[List[Parameter]]):
        super().__init__()
        self._module = module
        # TODO: shift the parameter level
        # find where the parameters live in the modules, install default
        # mapping
        shared_param_memo = {}
        self._indirections = {}
        for submodule_name, submodule in module.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                assert param not in shared_param_memo, "NYI"
                shared_param_memo[param] = (submodule, submodule_name, param_name)
                # gotta allocate fresh environments for the closures
                self._indirections[(submodule_name, param_name)] = (lambda param: lambda: param)(param)
        for param, memo in shared_param_memo.items():
            submodule, submodule_name, param_name = memo
            new_p = IndirectTensor((lambda s, p: lambda: self._indirections[(s, p)])(submodule_name, param_name))
            delattr(submodule, param_name)
            # TODO: make this look like a parameter
            setattr(submodule, param_name, new_p)
        # go through the buckets and update the mapping into the flat
        # parameters
        # TODO: shared params are not handled.  the aliasing should be detected
        # and the params coalesced into one location in the flat parameter
        # TODO: copying into a preallocated cat buffer save reallocation
        # TODO: this doesn't preserve memory format of the input parameters
        # TODO: check dtypes match
        for i, params in enumerate(param_buckets):
            flat_param = torch.cat([
                p.detach().clone(memory_format=torch.contiguous_format).view(-1)
                for p in params
            ], dim=0)
            self.register_buffer(f"flat_param{i}", flat_param)
            offset = 0
            for p in params:
                submodule, submodule_name, param_name = shared_param_memo[p]
                self._indirections[(submodule_name, param_name)] = \
                    flat_param[offset:offset + p.numel()].view(p.shape)
                offset += p.numel()

    def forward(self, *args, **kwargs):
        return self._module(*args, **kwargs)

model = nn.Sequential(
    nn.Linear(8, 4, bias=False),
    nn.Linear(4, 2, bias=False),
)

B = 10
input = torch.randn(B, 8)

print(model(input))

model = FlattenParamsWrapper(model, [[model[0].weight, model[1].weight]])
print(model.flat_param0)
print(type(model._module[0].weight))
print(model(input))
print(list(model.named_parameters()))
