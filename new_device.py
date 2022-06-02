import torch
from torch.overrides import enable_torch_function_mode, TorchFunctionMode
from torch.utils._pytree import tree_map

import numpy as np

aten = torch.ops.aten

# 1. A Tensor that stores custom raw_data and implement functions for it
class MyDeviceTensor(torch.Tensor):
    IMPLEMENTATIONS = {}

    @staticmethod
    def __new__(cls, size, dtype, raw_data=None, requires_grad=False):
        # Use a meta Tensor here to be used as the wrapper
        return torch.Tensor._make_subclass(
            cls,
            torch.empty(size, dtype=dtype, device="meta"),
            require_grad=requires_grad,
        )

    def __init__(self, size, dtype, raw_data=None, requires_grad=False):
        # Store any provided user raw_data
        self.raw_data = raw_data

    def __repr__(self):
        st = super().__repr__()
        st = st.replace("device='meta'", "device='my_device'")
        # Print the content the best way possible
        new_content = "[" + ", ".join(str(el) for el in self.raw_data) + "]"
        st = st.replace("...", new_content)
        return st

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func in cls.IMPLEMENTATIONS:
            try:

                def super_fn(*args, **kwargs):
                    return super(cls, MyDeviceTensor).__torch_dispatch__(
                        func, types, args, kwargs
                    )

                return cls.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})
            except Exception as e:
                print(e)
                raise e
        raise RuntimeError(
            f"No implementation for 'my_device' for {func}, {args}, {kwargs}"
        )


# Convenient wrapper to register functions
def implements(func):
    def _inner_fn(impl):
        MyDeviceTensor.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn


# Add some ops
@implements(aten.add.Tensor)
def add(super_fn, t1, t2):
    # You can do whatever you want with the raw data here
    # In particular, this can call any c++ code as needed.
    out = t1.raw_data + t2.raw_data
    return MyDeviceTensor(t1.size(), t1.dtype, raw_data=out)


@implements(aten.mul.Tensor)
def mul(super_fn, t1, t2):
    # If unsure what should be the result's properties, you can
    # use the super_fn (can be useful for type promotion)
    meta_out = super_fn(t1, t2)

    out = t1.raw_data * t2.raw_data
    return MyDeviceTensor(meta_out.size(), meta_out.dtype, raw_data=out)


# Add some trivial ops that need impl
@implements(aten.detach.default)
@implements(aten.alias.default)
def detach(super_fn, self):
    return super_fn(self)


# 2. A mode that allows us to override factory functions
# This needs to be a torch function mode before the arg parser creates a device
# based on the passed string, so we need to change it before reaching the arg parser
class MyDeviceMode(torch.Tensor):
    IMPLEMENTATIONS = {}

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        def super_fn(*args, **kwargs):
            # Disable torch_function by hand because we don't want the wrapping behavior of
            # the super() impl
            with torch._C.DisableTorchFunction():
                return func(*args, **kwargs)

        if func in cls.IMPLEMENTATIONS:
            try:
                return cls.IMPLEMENTATIONS[func](super_fn, *args, **kwargs or {})
            except Exception as e:
                print(e)
                raise e
        # This is just a no-op for all the non-factory functions:
        return super_fn(*args, **kwargs or {})


# Convenient wrapper to register functions
def implements_factory(func):
    def _inner_fn(impl):
        MyDeviceMode.IMPLEMENTATIONS[func] = impl
        return impl

    return _inner_fn


# Globally enable the mode
holder = enable_torch_function_mode(MyDeviceMode)
holder.__enter__()

# And some factory functions
# By hand
@implements_factory(torch.Tensor.to)
def to(super_fn, self, device):
    # Note that we only implement a subset of .to() here
    if device == "my_device":
        return MyDeviceTensor(self.size(), self.dtype, self.numpy())
    elif isinstance(self, MyDeviceTensor):
        return torch.from_numpy(self.raw_data).to(device)
    else:
        return super_fn(self, device)


# Have a nicer way to add many factories
def get_factory_wrapper(func):
    def inner(super_fn, size, **kwargs):
        if str(kwargs.get("device", None)) != "my_device":
            return super_fn(size, **kwargs)

        return MyDeviceTensor(size, kwargs.get("dtype", torch.float32), func(size))

    return inner


implements_factory(torch.rand)(get_factory_wrapper(np.random.rand))
implements_factory(torch.arange)(get_factory_wrapper(np.arange))
implements_factory(torch.empty)(get_factory_wrapper(np.empty))


if __name__ == "__main__":
    # 3. Show what it does in practice
    size = (2, 2)
    t1 = MyDeviceTensor(size, torch.float32, np.ones(size))
    t2 = MyDeviceTensor(size, torch.float32, np.arange(size[0] * size[1]).reshape(size))
    print("Inputs:")
    print(t1)
    print(t2)

    out = torch.add(t1, t2)
    print("torch.add(t1, t2):")
    print(out)

    out = t1 * t2
    print("t1 * t2:")
    print(out)

    # Factory functions
    t1 = torch.empty(4, device="my_device")
    print("Empty Tensor (un-initialized memory!):")
    print(t1)

    t1 = torch.rand(4, device="my_device")
    print("Random Tensor:")
    print(t1)

    t1 = torch.arange(4, device="my_device")
    print("Arange Tensor:")
    print(t1)

    t1 = torch.rand(5)
    print("Cpu Tensor:")
    print(t1)
    print("t2 = t1.to('my_device'):")
    t2 = t1.to("my_device")
    print(t2)
    print("t2.to('cpu'):")
    print(t2.to("cpu"))
