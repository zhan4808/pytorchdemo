"""C ABI shim for kernel stubs."""
import kernel_lib as kernels


def call_kernel(op_name, *args, **kwargs):
    """Dispatch a named kernel to the stub implementation."""
    if op_name == "gemm":
        return kernels.gemm(*args, **kwargs)
    if op_name == "relu":
        return kernels.relu(*args, **kwargs)
    if op_name == "softmax":
        return kernels.softmax(*args, **kwargs)
    raise NotImplementedError(f"Unsupported kernel op '{op_name}'")
