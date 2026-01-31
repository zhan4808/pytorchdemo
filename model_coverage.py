"""Export a few models and report op coverage."""
from dataclasses import dataclass
from typing import Iterable, List, Set

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


def _collect_ops(gm: torch.fx.GraphModule) -> List[str]:
    ops = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            ops.append(_target_to_str(node.target))
    return ops


def _report(name: str, ops: Iterable[str]) -> List[str]:
    ops_list = list(ops)
    missing = sorted({op for op in ops_list if op not in SUPPORTED_OPS})
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Ops ({len(ops_list)}): {ops_list}")
    if missing:
        print(f"Missing ({len(missing)}): {missing}")
    else:
        print("Missing (0): []")
    return missing


@dataclass
class ModelCase:
    name: str
    model: torch.nn.Module
    inputs: tuple


class MLP(torch.nn.Module):
    def forward(self, x, w1, w2):
        h = torch.relu(x @ w1)
        y = h @ w2
        return y


class AttentionBlock(torch.nn.Module):
    def forward(self, x, wq, wk, wv):
        q = x @ wq
        k = x @ wk
        v = x @ wv
        scores = q @ k.transpose(-2, -1)
        weights = torch.softmax(scores, dim=-1)
        return weights @ v


class ElementwiseMix(torch.nn.Module):
    def forward(self, a, b, c):
        return (a + b) * c - a


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


def main() -> None:
    cases = [
        ModelCase(
            "MLP",
            MLP(),
            (torch.randn(32, 64), torch.randn(64, 128), torch.randn(128, 32)),
        ),
        ModelCase(
            "AttentionBlock (no conv)",
            AttentionBlock(),
            (
                torch.randn(8, 32),
                torch.randn(32, 32),
                torch.randn(32, 32),
                torch.randn(32, 32),
            ),
        ),
        ModelCase(
            "ElementwiseMix",
            ElementwiseMix(),
            (torch.randn(16, 16), torch.randn(16, 16), torch.randn(16, 16)),
        ),
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

    all_missing: Set[str] = set()
    for case in cases:
        exported = torch.export.export(case.model, case.inputs)
        ops = _collect_ops(exported.graph_module)
        missing = _report(case.name, ops)
        all_missing.update(missing)
    if all_missing:
        raise SystemExit(f"Unsupported ops found: {sorted(all_missing)}")


if __name__ == "__main__":
    main()
