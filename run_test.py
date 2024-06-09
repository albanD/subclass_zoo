# Files that need to be run manually
files_to_run = [
    "autograd_monkeypatch",
    "cuda_sanitizer",
    "data_parallel_tensor",
    "deferred_init",
    "dynamic_shapes",
    "dynamic_strides",
    # "enhanced_error_mode",  # Actually raises an error
    "flat_view_tensor",
    "new_device",
    "py_dispatcher",
    "memory_debugging_tensor",
    "quantization_transform",
    "quantized_tensor",
    "simple_functorch",
    "torchdynamo_dynamic_inference",
    "tracing_guards",
    "use_cpu_for_rng",
]
cuda_only_files = {
    "cuda_sanitizer",
    "memory_debugging_tensor",
}

# Files with actual tests
import torch
from torch.testing._internal.common_utils import run_tests
from bug_zoo import BugZoo
from empty_tensor import EmptyTensorTest
from functorch_test import FunctorchTest
from inner_autograd_tensor import InnerAutogradTensorTest
from logging_mode import TracerTensorTest
from negative_tensor import NegativeTensorTest
# from nested_forward_ad import NestedForwardADTest
from progressive_lowering_tensor import ProgressiveLoweringTensorTest
from sparse_output import SparseOutputTest
from tracer_tensor import TracerTensorTest
from trivial_tensors import TrivialTensorTest
from verifier_tensor import VerifierTensorTest

if __name__ == "__main__":
    import os
    for file in files_to_run:
        print(f"Running {file}:")
        if (not torch.cuda.is_available()) and file in cuda_only_files:
            print("Skipped as cuda is not available")
            continue
        ret = os.system(f"python {file}.py 1> /dev/null 2>/dev/null")
        if ret != 0:
            print("Failure:")
            ret = os.system(f"python {file}.py")
            exit(-1)
        else:
            print("All good!")

    run_tests()
