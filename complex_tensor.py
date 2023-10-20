import torch


class ComplexTensor(torch.Tensor):
    def __new__(cls, re, im):
        assert (
            re.device == im.device
            and re.layout == im.layout
            and re.requires_grad == im.requires_grad
            and re.dtype == im.dtype
        )
        res = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            size=re.size(),
            strides=re.stride(),  # todo: contiguous only
            storage_offset=0,
            dtype=torch.complex64,  # todo: real to complex dtype
            layout=re.layout,
            device=re.device,
            requires_grad=False,  # todo: autograd support
        )
        res.re = re
        res.im = im
        return res

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mm.default:
            assert not kwargs
            x, y = args
            re = x.re * y.re - x.im * y.im
            im = x.re * y.im + x.im * y.re
            return ComplexTensor(re, im)
        raise NotImplementedError(f"todo {func}")

    def __tensor_flatten__(self):
        return ["re", "im"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        assert meta is None
        re, im = inner_tensors["re"], inner_tensors["im"]
        return ComplexTensor(re, im)

    def __repr__(self):
        return f"ComplexTensor(real={self.re}, imag={self.im})"


if __name__ == "__main__":

    @torch.compile()
    def f(x, y):
        return x @ y

    x = ComplexTensor(torch.tensor([[1]]), torch.tensor([[2]]))
    y = ComplexTensor(torch.tensor([[3]]), torch.tensor([[4]]))

    print(f(x, y))  # (1 + 2i) * (3 + 4i) = -5 + 10i
