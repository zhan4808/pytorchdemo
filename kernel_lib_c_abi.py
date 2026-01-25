"""
Simulated C ABI boundary for kernel calls.
In a real system this would be a C shared library invoked via ctypes/cffi.
"""
import kernel_lib as kernels


def call_kernel(op_name, *args, **kwargs):
    if op_name == "gemm":
        return kernels.gemm(*args, **kwargs)
    if op_name == "relu":
        return kernels.relu(*args, **kwargs)
    if op_name == "softmax":
        return kernels.softmax(*args, **kwargs)
    raise NotImplementedError(f"Unsupported kernel op '{op_name}'")
