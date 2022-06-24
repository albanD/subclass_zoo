from collections import defaultdict
from typing import Dict
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

MB = 1024 * 1024.0

operator_names: Dict[str, int] = defaultdict(int)
mem_usage: Dict[str, float] = defaultdict(float)
markers: Dict[str, int] = defaultdict(int)
series: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))


def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


def reduce_to_scalar_loss(inp):
    return inp.sum()


class MemoryProfileDispatchMode(TorchDispatchMode):
    def __init__(self, verbose=False):
        self.verbose: bool = verbose

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func == torch.ops.aten.detach.default:
            return rs
        mem: float = torch.cuda.memory_allocated() / MB
        func_name: str = func.__name__ + "_" + str(operator_names[func.__name__])
        operator_names[func.__name__] = operator_names[func.__name__] + 1
        mem_usage[func_name] = mem
        if self.verbose:
            print("Mem Usage (" + func_name + "): ", mem)
        return rs


def clear_state():
    global mem_usage, operator_names
    operator_names.clear()
    mem_usage.clear()


def add_series(series_name):
    global mem_usage
    fin_usage = torch.cuda.memory_allocated() / MB
    mem_usage["fin_usage"] = fin_usage
    series[series_name] = mem_usage
    mem_usage = defaultdict(float)


def save_graph(filename: str):

    for series_name, mem_usage in series.items():
        y = mem_usage.values()
        min_val = min(y)
        max_val = max(y)
        x = [i for i in range(len(y))]
        plt.plot(x, y, label=series_name)
    plt.xlabel("# Operator Calls")
    plt.ylabel("Allocated Memory (MB)")
    plt.title(filename)
    for marker_name, marker in markers.items():
        plt.plot([marker, marker], [min_val, max_val], "k-", lw=2, label=marker_name)
    plt.legend()
    print("Saving Graph")
    plt.savefig(filename)
    global series, markers
    series.clear()
    markers.clear()


def add_marker(marker_name):
    k = len(series.keys())
    last_val_num = len(mem_usage.values())
    markers[marker_name + str(k)] = last_val_num


def mem_profile_model(mod: torch.nn.Module, inp: torch.Tensor):

    with MemoryProfileDispatchMode(True):
        pred = mod(inp)
        loss = reduce_to_scalar_loss(pred)
        loss.backward()
        mod.zero_grad(True)
        torch.cuda.synchronize()
        clear_state()
        pred = mod(inp)
        loss = reduce_to_scalar_loss(pred)
        add_marker("fw_bw_boundary")
        loss.backward()


if __name__ == "__main__":
    try:
        import torchvision.models as models

        from functorch.compile import (
            aot_module,
            min_cut_rematerialization_partition,
            nop,
            print_compile,
        )

        mod: torch.nn.Module = models.resnet18().cuda()
        inp: torch.Tensor = torch.randn(32, 3, 224, 224, device="cuda")
        mem_profile_model(mod, inp)
        add_series("eager_mode")
        mod3 = aot_module(mod, nop, partition_fn=min_cut_rematerialization_partition)
        mem_profile_model(mod3, inp)
        add_series("aot_autograd_min_cut")
        save_graph("Resnet_mem_usage")
        clear_state()
        with MemoryProfileDispatchMode(True):
            mod3 = aot_module(
                mod, nop, partition_fn=min_cut_rematerialization_partition
            )
            mod3(inp).sum().backward()
            add_series("aot_autograd_mem_usage")
            save_graph("autograd_mem_usage")
    except ImportError:

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
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        mod: torch.nn.Module = MyModel().cuda()
        inp: torch.Tensor = torch.randn(512, 3, 32, 32, device="cuda")
        mem_profile_model(mod, inp)
        add_series("eager_mode")
        save_graph("Model_mem_usage")
