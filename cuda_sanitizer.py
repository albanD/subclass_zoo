import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

# TODO: dedupe from torch._subclasses.fake_tensor
def contains_tensor_types(type):
    tensor_type = torch._C.TensorType.get()
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )

class CUDASanitizer(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not kwargs:
            kwargs = {}

        # TODO: short circuit dispatch if no CUDA involved
        inputs = set()
        outputs = set()

        # TODO: a variant of tree map that also gives you the arg
        # schema would be pretty handy
        schema = func._schema
        for i, arg in enumerate(schema.arguments):
            if i < len(args):
                argument = args[i]
            else:
                if arg.name not in kwargs:
                    continue
                argument = kwargs[arg.name]
            if not contains_tensor_types(arg.type):
                continue
            mut_arg = False
            if arg.alias_info:
                if arg.alias_info.is_write:
                    mut_arg = True
            if isinstance(argument, torch.Tensor):
                if mut_arg:
                    outputs.add(argument.storage())
                else:
                    inputs.add(argument.storage())
            else:
                raise NotImplemented("todo tensor list")

        r = func(*args, **kwargs)

        def add_output(t):
            if isinstance(t, torch.Tensor):
                outputs.add(t.storage())

        tree_map(add_output, r)

        def render(storage):
            stream = torch.cuda.current_stream(storage.device)
            return f"ptr {storage.data_ptr():#08x} on stream {stream.cuda_stream:#08x}"

        readonly_str = ' '.join(map(render, inputs - outputs))
        readwrite_str = ' '.join(map(render, outputs))

        print(f"launch_kernel inputs {readonly_str} outputs {readwrite_str} # {schema}")
        return r

with CUDASanitizer.push():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        t = torch.ones((100,), device="cuda:0", requires_grad=True)

    with torch.cuda.stream(s2):
        s = t.sum()
        s.backward()
