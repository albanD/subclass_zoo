import torch

from base_tensor import BaseTensor
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)
from torch.utils._pytree import tree_map
from functools import partial
from torch.fx.operator_schemas import normalize_function
from typing import Union
import unittest

aten = torch.ops.aten


# Meta tensors give you the ability to run PyTorch code without having to
# actually do computation through tensors allocated on a `meta` device.
# Because the device is `meta`, meta tensors do not model device propagation.
# FakeTensor extends MetaTensors to also carry an additional `fake_device`
# which tracks devices that would have been used.


class FakeTensor(BaseTensor):
    fake_device: torch.device

    @staticmethod
    def __new__(cls, elem, device):
        return super().__new__(cls, elem)

    def __init__(self, elem, device: Union[torch.device, str]):
        # elem does not need to be recorded, because FakeTensor *is a* elem
        assert elem.device.type == "meta"
        device if isinstance(device, torch.device) else torch.device(device)
        assert device.type != "meta"
        self.fake_device = device

    @staticmethod
    def from_tensor(t):
        existing_device = t.device
        return FakeTensor(t.to(device="meta"), existing_device)

    @property
    def device(self):
        return self.fake_device

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # Run the original computation
        r = super().__torch_dispatch__(func, types, args, kwargs)
        kwargs = kwargs if kwargs else {}

        def wrap(e, device):
            # inplace ops can return fake tensors
            if isinstance(e, torch.Tensor) and not isinstance(e, cls):
                return FakeTensor(e, device)
            else:
                return e

        # Pytorch device is kwarg-only, except for _pin_memory/pin_memory
        # which is not yet supported by meta-tensors
        assert (
            func != aten._pin_memory.default and func != aten.pin_memory.default
        ), f"NYI: {func}"

        # if device is specified, use that
        # not sure this is actually needed.. device only shows up in constructors
        if kwargs.get("device", None):
            return tree_map(partial(wrap, device=kwargs["device"]), r)

        # operators which copy size from another tensor do not
        # also take device from the size tensor
        # other size_as operators are not builtin operators
        if func == torch.ops.aten.resize_as_.default:
            # TODO: https://github.com/pytorch/pytorch/pull/77182
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            # device of the input is returned
            return tree_map(partial(wrap, device=new_kwargs["input"].device), r)

        # TODO, file issue, this path does not get exercised
        if func is torch.ops.aten.type_as:
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            return tree_map(partial(wrap, device=new_kwargs["other"].device), r)

        def cpu_zero_dim(t):
            return t.device == torch.device("cpu") and t.dim() == 0

        # cpu - zero-dim tensors can be called in cuda kernels,
        # so overwrite cuda kernels
        common_device = None
        is_cpu_zero_dim = None

        def find_common_device(t):
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            if not isinstance(t, cls):
                return

            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            t_is_cpu_zero_dim = cpu_zero_dim(t)
            if t.device == common_device:
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices !
            # if current tensor is cpu 0 dim, defer to existing device
            if t_is_cpu_zero_dim:
                return

            # current device is from cpu 0 dim tensor, overwrite
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = t_is_cpu_zero_dim
                return

            # mismatching devices of non-zero dim tensors, throw
            # This might be valid behavior and need to be explicitly modeled, e.g. reshape_as
            raise Exception(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        tree_map(find_common_device, args)
        tree_map(find_common_device, kwargs)

        assert common_device != None, f"Could not find common device for {func}"

        return tree_map(partial(wrap, device=common_device), r)


class FakeTensorTest(TestCase):
    def test_basic(self):
        x = FakeTensor.from_tensor(torch.empty(2, 2, device="cpu"))
        y = x = FakeTensor.from_tensor(torch.empty(4, 2, 2, device="cpu"))
        y = x + x
        self.assertEqual(y.shape, (4, 2, 2))
        self.assertEqual(y.device, torch.device("cpu"))

    @unittest.skip("Waiting on https://github.com/pytorch/pytorch/pull/77182")
    def test_shape_take_not_device(self):
        x = FakeTensor.from_tensor(torch.empty(1, device="cpu"))
        y = FakeTensor.from_tensor(torch.empty(8, 8, device="cuda"))
        out = x.resize_as_(y)
        self.assertEqual(out.shape, (8, 8))
        self.assertEqual(out.device, torch.device("cpu"))

    def test_zero_dim(self):
        x = FakeTensor.from_tensor(torch.tensor(0.0))
        y = FakeTensor.from_tensor(torch.rand([4, 4], device="cuda"))
        out = x + y
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(out.device, y.device)

    def test_throw(self):
        x = FakeTensor.from_tensor(torch.tensor(0.0))
        y = FakeTensor.from_tensor(torch.rand([4, 4], device="cuda"))
        z = FakeTensor.from_tensor(torch.rand([4, 4], device="cpu"))
        self.assertRaises(
            Exception, lambda: torch.lerp(x, y, z)
        )

if __name__ == "__main__":
    run_tests()
