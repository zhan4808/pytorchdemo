"""
Prototype of a custom accelerator backend for torch.compile.
Demonstrates the dispatch flow from FX graph to op implementations.
"""
from typing import List

import torch
from torch.fx import GraphModule

import kernel_lib as kernels


def my_accel_matmul(a, b):
    print(f"  [MY_ACCEL] matmul called: {a.shape} @ {b.shape}")
    return kernels.gemm(a, b)


def my_accel_relu(x):
    print(f"  [MY_ACCEL] relu called: {x.shape}")
    return kernels.relu(x)


def my_accel_softmax(x, dim):
    print(f"  [MY_ACCEL] softmax called: {x.shape}, dim={dim}")
    return kernels.softmax(x, dim=dim)


OP_REGISTRY = {
    torch.matmul: my_accel_matmul,
    torch.relu: my_accel_relu,
    torch.nn.functional.relu: my_accel_relu,
    torch.softmax: my_accel_softmax,
    torch.nn.functional.softmax: my_accel_softmax,
}


def my_accel_backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """
    Custom backend for a prototype accelerator.
    """
    print("\n" + "=" * 60)
    print("MY_ACCEL BACKEND - GRAPH RECEIVED")
    print("=" * 60)

    ops_found = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            ops_found.append(node.target)
            supported = "✓ SUPPORTED" if node.target in OP_REGISTRY else "✗ NOT SUPPORTED (will fallback)"
            name = node.target.__name__ if hasattr(node.target, "__name__") else str(node.target)
            print(f"  Found op: {name} - {supported}")

    print(f"\nTotal ops: {len(ops_found)}")
    print(f"Supported: {sum(1 for op in ops_found if op in OP_REGISTRY)}")

    def custom_executor(*args):
        print("\n--- EXECUTING ON MY_ACCEL ---")

        env = {}
        arg_iter = iter(args)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                env[node.name] = next(arg_iter)

        for node in gm.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "call_function":
                fn_args = [env[a.name] if hasattr(a, "name") else a for a in node.args]
                fn_kwargs = {
                    k: env[v.name] if hasattr(v, "name") else v
                    for k, v in node.kwargs.items()
                }

                if node.target in OP_REGISTRY:
                    result = OP_REGISTRY[node.target](*fn_args, **fn_kwargs)
                else:
                    print(f"  [FALLBACK] {node.target}")
                    result = node.target(*fn_args, **fn_kwargs)

                env[node.name] = result
            elif node.op == "output":
                output_nodes = node.args[0]
                if isinstance(output_nodes, tuple):
                    return tuple(env[n.name] for n in output_nodes)
                return env[output_nodes.name]

        return None

    return custom_executor


@torch.compile(backend=my_accel_backend)
def test_model(x, w):
    h = torch.relu(x)
    y = torch.matmul(h, w)
    z = torch.softmax(y, dim=-1)
    return z


def main():
    print("Testing custom accelerator backend prototype\n")

    x = torch.randn(4, 8)
    w = torch.randn(8, 16)

    output = test_model(x, w)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Output sum: {output.sum():.4f}")

    print("\n--- Second call (should use cached graph) ---")
    output2 = test_model(torch.randn(4, 8), w)
    print(f"Output2 sum: {output2.sum():.4f}")


if __name__ == "__main__":
    main()
