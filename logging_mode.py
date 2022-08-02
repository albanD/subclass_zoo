import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from torch._subclasses.meta_utils import WeakTensorRefKey
from torch.testing._internal.common_utils import run_tests, TestCase
import torch.overrides

import weakref
from functools import partial
import itertools



dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}


class Lit:
    def __init__(self, s):
        self.s = s

    def __repr__(self):
        return self.s


class LoggingMode(TorchDispatchMode):
    next_id: int

    def __init__(self, with_type: bool = True, collect_logs=False):
        self.memo = {}
        self.next_id = 0
        self.with_type = with_type
        self.collect_logs = collect_logs
        self.logs = []

    def _shortid(self, t: torch.Tensor) -> int:
        o = WeakTensorRefKey(t)
        weak_self = weakref.ref(self)

        def del_memo():
            self = weak_self()
            if self is None:
                return
            self.memo.pop(o, None)

        weakref.finalize(t, del_memo)
        if o not in self.memo:
            self.memo[o] = self.next_id
            self.next_id += 1
        return self.memo[o]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        if isinstance(a, torch.Tensor):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            return Lit(f"${self._shortid(a)}{maybe_type}")
        else:
            return a

    def str_logs(self):
        return '\n'.join(self.logs)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        fmt_args = ", ".join(
            itertools.chain(
                (repr(tree_map(self._fmt, a)) for a in args),
                (f"{k}={tree_map(self._fmt, v)}" for k, v in kwargs.items()),
            )
        )
        fmt_rets = repr(tree_map(partial(self._fmt, with_type=True), rs))
        log_msg = f"{fmt_rets} = {torch.overrides.resolve_name(func)}({fmt_args})"
        if self.collect_logs:
            self.logs.append(log_msg)
        else:
            print(log_msg)
        return rs


with LoggingMode():
    torch.nn.functional.dropout(torch.randn(3), 0.5)


class TracerTensorTest(TestCase):
    def test_basic(self):
        with LoggingMode(collect_logs=True) as mode:
            x = torch.randn(2, 3, requires_grad=True)
            y = torch.randn(3, 4)
            with torch.autocast('cpu'):
                r = x @ y
                r.sum().backward()
        self.assertExpectedInline(
            mode.str_logs(),
            """\
$0: f32[2, 3] = aten.randn.default([2, 3], dtype=torch.float32, device=cpu, pin_memory=False)
$1: f32[3, 4] = aten.randn.default([3, 4], dtype=torch.float32, device=cpu, pin_memory=False)
$2: bf16[3, 4] = aten._to_copy.default($1, dtype=torch.bfloat16)
$3: bf16[2, 3] = aten._to_copy.default($0, dtype=torch.bfloat16)
$4: bf16[2, 4] = aten.mm.default($3, $2)
$5: bf16[] = aten.sum.default($4)
$6: bf16[] = aten.ones_like.default($5, dtype=torch.bfloat16, layout=torch.strided, device=cpu, pin_memory=False, memory_format=torch.preserve_format)
$7: bf16[2, 4] = aten.expand.default($6, [2, 4])
$8: bf16[4, 3] = aten.t.default($2)
$9: bf16[2, 3] = aten.mm.default($7, $8)
$10: f32[2, 3] = aten._to_copy.default($9, dtype=torch.float32, layout=torch.strided, device=cpu)
$11: f32[2, 3] = aten.detach.default($10)""",  # noqa
        )

if __name__ == "__main__":
    run_tests()
