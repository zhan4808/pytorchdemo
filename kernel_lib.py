"""Kernel stubs used by the demo runtime."""
import torch


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """GEMM stub using PyTorch matmul."""
    return a @ b


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU stub using max with zero."""
    return torch.maximum(x, torch.zeros_like(x))


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax stub with a stable max-subtract."""
    x_max = x.max(dim=dim, keepdim=True).values
    exp = torch.exp(x - x_max)
    denom = exp.sum(dim=dim, keepdim=True)
    return exp / denom


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise add stub."""
    return a + b


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise sub stub."""
    return a - b


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise mul stub."""
    return a * b


def transpose(x: torch.Tensor, dim0: int, dim1: int) -> torch.Tensor:
    """Transpose stub using PyTorch transpose."""
    return x.transpose(dim0, dim1)
