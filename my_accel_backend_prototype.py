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


def _encode_arg(arg):
    if hasattr(arg, "name"):
        return ("node", arg.name)
    return ("lit", arg)


def _decode_arg(encoded, env):
    kind, value = encoded
    if kind == "node":
        return env[value]
    return value


def compile_graph(gm: GraphModule):
    """
    Lower an FX graph into a tiny "program" for a runtime to execute.
    This models a compiler that emits a backend-specific instruction list.
    """
    program = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            program.append({"op": "placeholder", "name": node.name})
        elif node.op == "call_function":
            program.append(
                {
                    "op": "call_function",
                    "name": node.name,
                    "target": node.target,
                    "supported": node.target in OP_REGISTRY,
                    "args": [_encode_arg(a) for a in node.args],
                    "kwargs": {k: _encode_arg(v) for k, v in node.kwargs.items()},
                }
            )
        elif node.op == "output":
            output_nodes = node.args[0]
            if isinstance(output_nodes, tuple):
                names = [n.name for n in output_nodes]
            else:
                names = [output_nodes.name]
            program.append({"op": "output", "names": names})
        else:
            program.append(
                {
                    "op": "unsupported",
                    "node_op": node.op,
                    "target": node.target,
                    "name": node.name,
                }
            )
    return program


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

    program = compile_graph(gm)
    call_function_count = sum(1 for instr in program if instr["op"] == "call_function")
    print(f"\nCompiling graph -> {call_function_count} call_function nodes")
    print(f"Program length: {len(program)}")

    def custom_executor(*args):
        print("\n--- EXECUTING ON MY_ACCEL ---")

        env = {}
        arg_iter = iter(args)

        for instr in program:
            if instr["op"] == "placeholder":
                env[instr["name"]] = next(arg_iter)
            elif instr["op"] == "call_function":
                fn_args = [_decode_arg(a, env) for a in instr["args"]]
                fn_kwargs = {k: _decode_arg(v, env) for k, v in instr["kwargs"].items()}

                if instr["supported"]:
                    result = OP_REGISTRY[instr["target"]](*fn_args, **fn_kwargs)
                else:
                    print(f"  [FALLBACK] {instr['target']}")
                    result = instr["target"](*fn_args, **fn_kwargs)

                env[instr["name"]] = result
            elif instr["op"] == "output":
                names = instr["names"]
                if len(names) == 1:
                    return env[names[0]]
                return tuple(env[name] for name in names)
            elif instr["op"] == "unsupported":
                raise NotImplementedError(
                    f"Unsupported node op '{instr['node_op']}' for target '{instr['target']}'"
                )

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
