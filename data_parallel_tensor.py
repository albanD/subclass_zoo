from enum import Enum, auto
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._C import NoneType, device
from torch._utils import _get_all_device_indices
from torch.cuda import comm
from torch.utils._pytree import tree_map

torch.__future__.set_overwrite_module_params_on_conversion(True)
import concurrent.futures as futures

_ = torch.manual_seed(0)
aten = torch.ops.aten
NUM_DEVICES = 8
PARALLEL_DISPATCH = False
ALL_REDUCE = True


class DPTensorType(Enum):
    replicated = auto()  # This tensor will be replicated across all the devices
    distributed_batch = (
        auto()
    )  # This tensor will be sharded along the first/batch dimension across
    # the devices, NOTE: only equal chunk sizes are supported
    distributed = (
        auto()
    )  # This is a list of tensors, each of which rests on different devices


class DataParallelTensor(torch.Tensor):
    # This class is a tensor subclass that stores a list of tensors with the aim
    # DataParallelTensors(DPT) are categorized in three ways
    # 1) replicated: When a single tensor is supplied, it is replicated across
    #   all the devices by using broadcast
    # 2) distributed: DPT can also be initialized by supplying a list/tuple of tensors
    #   if the elements rest on different devices, they will just be wrapped in DPT
    #   else the elements are scattered to different devices
    # 3) distributed batch: This type of DPT tensor is created by sharding the input tensor across
    #   a specified sharding dimension (default: 0). Currently only equal chunk sizes are supported.

    elem: List[torch.Tensor]

    if torch.cuda.is_available():
        # device_ids: List[int] = _get_all_device_indices()
        device_ids = [i for i in range(NUM_DEVICES)]
        if PARALLEL_DISPATCH:
            num_threads: int = len(device_ids)
            threadpool: futures.ThreadPoolExecutor = futures.ThreadPoolExecutor(
                max_workers=num_threads
            )
    __slots__ = ["elem"]

    @staticmethod
    def __new__(
        cls,
        elem: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        func: Optional[Any] = None,
        dpt_type: DPTensorType = DPTensorType.replicated,
        batch_dim: Optional[int] = 0,
    ):

        if dpt_type == DPTensorType.replicated:
            # NOTE: If the input is None, we return None
            if elem is None:
                return None
            assert isinstance(elem, torch.Tensor)
            # NOTE: For handling meta tensors, if the device of an input tensor is meta,
            # we just return the first element in such a list/tuple
            if elem.device == device("meta"):
                return elem

            with torch.no_grad():
                dpt: List[torch.Tensor] = comm.broadcast(elem, devices=cls.device_ids)

        elif dpt_type == DPTensorType.distributed:
            assert isinstance(elem, list) or isinstance(elem, tuple)
            # NOTE: For handling None values, if any element of a list/tuple is None, we return None
            check_none = [True if e is None else False for e in elem]
            if any(check_none):
                return None
            check_not_tensor = [
                True if not isinstance(e, torch.Tensor) else False for e in elem
            ]
            if any(check_not_tensor):
                # NOTE: Need to define behaviour when an operation returns a tuple/list of vlaues that are not tensors
                # Currently we just return the first elemt of such a list/tuple
                return elem[0]
            requires_scatter: bool = False
            with torch.no_grad():
                for t, d_id in zip(elem, cls.device_ids):
                    if t.device == device("meta"):
                        # NOTE: For handling meta tensors, if the device of any tensor in such a list/tuple is meta,
                        # we just return the first element in such a list/tuple
                        return elem[0]
                    if t.device != device(d_id):
                        requires_scatter = True
                        break

                if requires_scatter:
                    # We first stack all the tensors in the list/tuple along dimension 0, to get a single tensor
                    # We then scatter the tensor along the 0th dimension to different devices
                    # The scatter function returns a list of tensors with a redundant 0th dimension for each element
                    # We squeeze out the redundant dimension from each of these elements to finally get a list of tensors
                    # each residing on a list of devices
                    stacked_t: torch.Tensor = torch.stack(elem, dim=0)
                    scattered_t: Tuple[torch.Tensor] = comm.scatter(
                        stacked_t, devices=cls.device_ids, dim=0
                    )
                    dpt: List[torch.Tensor] = [
                        torch.squeeze(t, dim=0) for t in scattered_t
                    ]
                else:
                    dpt: List[torch.Tensor] = elem
        elif dpt_type == DPTensorType.distributed_batch:
            # NOTE: This requires the batch dimension to be divisible by the number of devices.
            assert isinstance(elem, torch.Tensor)

            with torch.no_grad():
                scattered_t: Tuple[torch.Tensor] = comm.scatter(
                    elem, devices=cls.device_ids, dim=batch_dim
                )
                dpt: List[torch.Tensor] = list(scattered_t)

        meta_t: torch.Tensor = elem if dpt_type == DPTensorType.replicated else dpt[0]

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            meta_t.size(),
            strides=meta_t.stride(),
            storage_offset=meta_t.storage_offset(),
            device=meta_t.device,  # This is the device of of either input tensor or first tensor of a list
            dtype=meta_t.dtype,
            layout=meta_t.layout,
            requires_grad=meta_t.requires_grad,
        )
        r.elem = dpt
        return r

    def __repr__(self):
        if self.grad_fn:
            return f"DataParallelTensor({self.elem}, grad_fn={self.grad_fn})"
        return f"DataParallelTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def wrap(e):
            if isinstance(e, DataParallelTensor):
                return e
            elif isinstance(e, torch.Tensor):
                return DataParallelTensor(e, func, DPTensorType.replicated)
            else:
                return e

        # All the args and kwargs are checked and any leaf tensors are wrapped as replicated DPTs
        args = tree_map(wrap, args)
        kwargs = tree_map(wrap, kwargs)

        def unwrap_with_position(pos):
            def get_element(e):
                return e.elem[pos] if isinstance(e, DataParallelTensor) else e

            return get_element

        # Call the function for each of the DPT elements by unwarpping them and corresponding args and kwargs,
        #  into element tensors so that the operation is performed on all the elements residing on the same device
        if PARALLEL_DISPATCH:
            future_res: List[futures.Future] = []
            for pos in range(cls.num_threads):
                future_res.append(
                    cls.threadpool.submit(
                        func,
                        *tree_map(unwrap_with_position(pos), args),
                        **tree_map(unwrap_with_position(pos), kwargs),
                    )
                )
            outs = [future_res[i].result() for i in range(cls.num_threads)]
        else:
            outs = []
            for pos in range(len(cls.device_ids)):
                outs.append(
                    func(
                        *tree_map(unwrap_with_position(pos), args),
                        **tree_map(unwrap_with_position(pos), kwargs),
                    )
                )

        def get_element_type(lis):
            assert isinstance(lis, list)
            return type(lis[0])

        # The ouput will always be a list
        # The list can contain tensors, bools, list of tensors or tuples of tensors or None
        # In case of tensors we just wrap them in DPT
        # In case of list/tuple of tensors, the corresponding elements across list/tuple are warpped
        #  into a DPT and a list/tuple is returned respectively

        def out_wrap(e, func):
            elem_type = get_element_type(e)
            if elem_type is NoneType:
                return None
            if elem_type == torch.Tensor:
                return DataParallelTensor(outs, func, DPTensorType.distributed)
            elif elem_type == list:
                return list(
                    DataParallelTensor(list(t), func, DPTensorType.distributed)
                    for t in zip(*e)
                )
            elif elem_type == tuple:
                return tuple(
                    DataParallelTensor(list(t), func, DPTensorType.distributed)
                    for t in zip(*e)
                )
            elif elem_type == bool:
                return all(e)
            else:
                # NOTE: Think about handling this
                print("Warning...")
                return e[0]

        outs = out_wrap(outs, func)
        return outs

    def all_reduce_grad(
        self,
        r_device: Optional[int] = torch.cuda.current_device()
        if torch.cuda.is_available()
        else 0,
    ):
        with torch.no_grad():
            reduced_tensor: torch.Tensor = comm.reduce_add(self.elem, r_device)
            b_tensor: List[torch.Tensor] = comm.broadcast(reduced_tensor, out=self.elem)
            self.elem = b_tensor
        return reduced_tensor


def make_data_parallel_module(mod: torch.nn.Module):
    # This function converts the parameters of a nn.Module to replicated DataParallelTensors
    # the else part is important for buffers of the module
    def wrapper(t):
        if isinstance(t, torch.nn.Parameter):
            return DataParallelTensor(t.data, None, DPTensorType.replicated)
        else:
            assert type(t) in (torch.Tensor, NoneType, bool)
            return DataParallelTensor(t, None, DPTensorType.replicated)

    mod._apply(wrapper)


if __name__ == "__main__":

    if torch.cuda.is_available():
        print("Devices: ", [i for i in range(NUM_DEVICES)])
    else:
        print("Need GPUs to run examples")
        exit()

    try:
        from functools import partial

        from functorch import hessian, jacfwd, jacrev, vjp, vmap

        D = 16
        x: torch.Tensor = torch.randn(D, device="cuda")
        dpt_x = DataParallelTensor(x, None, DPTensorType.replicated)

        def predict(weight, bias, x):
            return F.linear(x, weight, bias).tanh()

        weight = torch.randn(D, D, device="cuda")
        bias = torch.randn(D, device="cuda")

        # Computing Jacobian using vmap and vjp and jacrev
        clone_x = dpt_x.clone().requires_grad_()
        unit_vectors = torch.eye(D).cuda()

        _, vjp_fn = vjp(partial(predict, weight, bias), clone_x)
        (ft_jacobian,) = vmap(vjp_fn)(unit_vectors)

        clone_x = dpt_x.clone().requires_grad_()
        jacobian_rev = jacrev(predict, argnums=2)(weight, bias, clone_x)

        print(torch.allclose(ft_jacobian, jacobian_rev))

        # Computing Hessian using composition of jacrev and jacfwd vs hessian api
        clone_x = dpt_x.clone().requires_grad_()
        hess_api = hessian(predict, argnums=2)(weight, bias, clone_x)
        hess_fwdrev = jacfwd(jacrev(predict, argnums=2), argnums=2)(
            weight, bias, clone_x
        )
        print(torch.allclose(hess_api, hess_fwdrev))
    except ImportError:
        print("Skipping functorch example, package missing.")

    try:
        # Example with a torchvision model
        import torchvision.models as models

        batch_size = 256
        test_tensor: torch.Tensor = torch.randn(
            batch_size * NUM_DEVICES, 3, 224, 224, device="cuda"
        )
        dp_tensor = DataParallelTensor(
            test_tensor, None, DPTensorType.distributed_batch
        )
        model = models.resnet50().cuda()
        make_data_parallel_module(model)
        # Warmp up iteration
        out = model(dp_tensor)
        loss = out.sum()
        loss.backward()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for i in range(1):
            out = model(dp_tensor)
            loss = out.sum()
            loss.backward()
            if ALL_REDUCE:
                for p in model.parameters():
                    p.grad.all_reduce_grad()
            #     p = p - 0.5 * p.grad
        end_event.record()
        torch.cuda.synchronize()
        print("Timing for 1 iteration (ms) DPT: ", start_event.elapsed_time(end_event))

        test_tensor: torch.Tensor = torch.randn(batch_size, 3, 224, 224, device="cuda")
        model = models.resnet50().cuda()
        # Warmp up iteration
        out = model(test_tensor)
        loss = out.sum()
        loss.backward()
        start_event.record()
        for i in range(NUM_DEVICES):
            out = model(test_tensor)
            loss = out.sum()
            loss.backward()

        # for p in model.parameters():
        #     p = p - 0.5 * p.grad

        end_event.record()
        torch.cuda.synchronize()
        print(
            "Timing for " + str(NUM_DEVICES) + " iterations(ms): ",
            start_event.elapsed_time(end_event),
        )
    except ImportError:
        print("Running custom model since torchvision package is absent.")

        # Custom Model Example
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 10)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        mod: torch.nn.Module = MyModel().cuda()
        inp: torch.Tensor = torch.randn(512, 3, 32, 32, device="cuda")
        dpt_inp = DataParallelTensor(inp, None, DPTensorType.distributed_batch)
        make_data_parallel_module(mod)
        out = mod(dpt_inp)
        loss = out.sum()
        loss.backward()

        for p in mod.parameters():
            p.grad.all_reduce_grad()
            p = p - 0.5 * p.grad

    # Custom Function Example
    test_tensor = torch.randn(8, 5, device="cuda", requires_grad=True)
    dp_tensor = DataParallelTensor(test_tensor, None, DPTensorType.distributed_batch)

    def custom_func(x):
        return x.cos().cos().sum()

    res_tensor = custom_func(dp_tensor)
    print(res_tensor)
    res_tensor.backward()
    print(dp_tensor.grad)
