"""Pipeline spec checks for Atalla Phase 1."""
from dataclasses import dataclass
from typing import Iterable, List

import torch


SUPPORTED_OPS = {
    "aten.matmul.default",
    "aten.relu.default",
    "aten.softmax.int",
    "aten.softmax.default",
    "aten.add.Tensor",
    "aten.sub.Tensor",
    "aten.mul.Tensor",
    "aten.transpose.int",
}


def _target_to_str(target) -> str:
    return str(target)


def _is_row_major_contiguous(t: torch.Tensor) -> bool:
    if not t.is_contiguous():
        return False
    if t.ndim == 0:
        return True
    expected = [1]
    for size in reversed(t.shape[1:]):
        expected.insert(0, expected[0] * size)
    return tuple(expected) == t.stride()


def _extract_softmax_dim(node) -> int:
    if "dim" in node.kwargs:
        return int(node.kwargs["dim"])
    if len(node.args) >= 2:
        return int(node.args[1])
    return -1


def _extract_transpose_dims(node) -> List[int]:
    dims = []
    if len(node.args) >= 3:
        dims = [int(node.args[1]), int(node.args[2])]
    elif "dim0" in node.kwargs and "dim1" in node.kwargs:
        dims = [int(node.kwargs["dim0"]), int(node.kwargs["dim1"])]
    return dims


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim = ndim + dim
    return dim


def _check_ops(gm: torch.fx.GraphModule) -> List[str]:
    missing = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target = _target_to_str(node.target)
        if target not in SUPPORTED_OPS:
            missing.append(target)
        if target == "aten.transpose.int":
            dims = _extract_transpose_dims(node)
            if len(dims) != 2:
                raise ValueError("Transpose dims not found")
    return missing


@dataclass
class ModelCase:
    name: str
    model: torch.nn.Module
    inputs: tuple


class TinyTransformerBlock(torch.nn.Module):
    def forward(self, x, wq, wk, wv, wo, w1, w2):
        q = x @ wq
        k = x @ wk
        v = x @ wv
        scores = q @ k.transpose(-2, -1)
        weights = torch.softmax(scores, dim=-1)
        attn = weights @ v
        proj = attn @ wo
        hidden = torch.relu(proj @ w1)
        out = hidden @ w2
        return out + x


def _check_inputs(inputs: Iterable[torch.Tensor]) -> None:
    for idx, t in enumerate(inputs):
        if not _is_row_major_contiguous(t):
            raise ValueError(f"Input {idx} is not row-major contiguous")


def _validate_runtime_constraints(gm: torch.fx.GraphModule, inputs: tuple) -> None:
    env = {}
    arg_iter = iter(inputs)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            env[node.name] = next(arg_iter)
            continue
        if node.op == "call_function":
            args = [
                env[a.name] if hasattr(a, "name") else a for a in node.args
            ]
            kwargs = {
                k: (env[v.name] if hasattr(v, "name") else v)
                for k, v in node.kwargs.items()
            }
            target = _target_to_str(node.target)
            if target.startswith("aten.softmax"):
                dim = _extract_softmax_dim(node)
                x = args[0]
                last_dim = x.ndim - 1
                dim = _normalize_dim(dim, x.ndim)
                if dim != last_dim:
                    raise ValueError(
                        f"Softmax dim must be last, got {dim} for shape {x.shape}"
                    )
            if target == "aten.transpose.int":
                dims = _extract_transpose_dims(node)
                if len(dims) == 2:
                    x = args[0]
                    d0 = _normalize_dim(dims[0], x.ndim)
                    d1 = _normalize_dim(dims[1], x.ndim)
                    if d0 < 0 or d1 < 0 or d0 >= x.ndim or d1 >= x.ndim:
                        raise ValueError(
                            f"Transpose dims out of range: {dims} for shape {x.shape}"
                        )
            env[node.name] = node.target(*args, **kwargs)
            continue
        if node.op == "output":
            break


def main() -> None:
    cases = [
        ModelCase(
            "TinyTransformerBlock (no conv)",
            TinyTransformerBlock(),
            (
                torch.randn(8, 32),
                torch.randn(32, 32),
                torch.randn(32, 32),
                torch.randn(32, 32),
                torch.randn(32, 32),
                torch.randn(32, 64),
                torch.randn(64, 32),
            ),
        ),
    ]

    for case in cases:
        _check_inputs(case.inputs)
        exported = torch.export.export(case.model, case.inputs)
        missing = _check_ops(exported.graph_module)
        if missing:
            raise SystemExit(f"{case.name}: unsupported ops {sorted(set(missing))}")
        _validate_runtime_constraints(exported.graph_module, case.inputs)
        print(f"{case.name}: OK")


if __name__ == "__main__":
    main()
