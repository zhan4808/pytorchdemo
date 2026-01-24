"""
Minimal "kernel" library implemented in Python.
These are functional placeholders for real C/C++/ISA kernels.
"""
import torch


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    GEMM kernel placeholder.
    """
    return a @ b


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ReLU kernel placeholder.
    """
    return torch.maximum(x, torch.zeros_like(x))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax kernel placeholder.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    denom = exp.sum(dim=dim, keepdim=True)
    return exp / denom
