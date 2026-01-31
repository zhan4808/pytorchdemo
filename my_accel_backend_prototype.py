"""Prototype backend: export → op registry → runtime."""
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


def my_accel_add(a, b):
    print(f"  [MY_ACCEL] add called: {a.shape} + {b.shape}")
    return kernels.add(a, b)


def my_accel_sub(a, b):
    print(f"  [MY_ACCEL] sub called: {a.shape} - {b.shape}")
    return kernels.sub(a, b)


def my_accel_mul(a, b):
    print(f"  [MY_ACCEL] mul called: {a.shape} * {b.shape}")
    return kernels.mul(a, b)


def my_accel_transpose(x, dim0, dim1):
    print(f"  [MY_ACCEL] transpose called: {x.shape}, dim0={dim0}, dim1={dim1}")
    return kernels.transpose(x, dim0=dim0, dim1=dim1)


OP_REGISTRY = {
    "aten.matmul.default": my_accel_matmul,
    "aten.relu.default": my_accel_relu,
    "aten.softmax.int": my_accel_softmax,
    "aten.softmax.default": my_accel_softmax,
    "aten.add.Tensor": my_accel_add,
    "aten.sub.Tensor": my_accel_sub,
    "aten.mul.Tensor": my_accel_mul,
    "aten.transpose.int": my_accel_transpose,
}


def _target_to_str(target):
    return str(target)


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
    """Lower an FX graph into a linear instruction list."""
    program = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            program.append({"op": "placeholder", "name": node.name})
        elif node.op == "call_function":
            target_str = _target_to_str(node.target)
            program.append(
                {
                    "op": "call_function",
                    "name": node.name,
                    "target": target_str,
                    "supported": target_str in OP_REGISTRY,
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
    """Compile a graph and return a runtime callable."""
    print("\n" + "=" * 60)
    print("MY_ACCEL BACKEND - GRAPH RECEIVED")
    print("=" * 60)

    ops_found = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_str = _target_to_str(node.target)
            ops_found.append(target_str)
            supported = "✓ SUPPORTED" if target_str in OP_REGISTRY else "✗ NOT SUPPORTED"
            print(f"  Found op: {target_str} - {supported}")

    print(f"\nTotal ops: {len(ops_found)}")
    print(f"Supported: {sum(1 for op in ops_found if op in OP_REGISTRY)}")
    unsupported = sorted({op for op in ops_found if op not in OP_REGISTRY})
    if unsupported:
        raise NotImplementedError(
            "Unsupported ops in accelerator-only mode: "
            + ", ".join(unsupported)
        )

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

                if not instr["supported"]:
                    raise NotImplementedError(
                        f"Unsupported op '{instr['target']}' in accelerator-only mode"
                    )
                result = OP_REGISTRY[instr["target"]](*fn_args, **fn_kwargs)

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


class TestModel(torch.nn.Module):
    def forward(self, x, w):
        h = torch.relu(x)
        y = torch.matmul(h, w)
        z = torch.softmax(y, dim=-1)
        return z


def main():
    print("Testing custom accelerator backend prototype\n")

    x = torch.randn(4, 8)
    w = torch.randn(8, 16)

    model = TestModel()
    exported = torch.export.export(model, (x, w))
    gm = exported.graph_module
    runtime = my_accel_backend(gm, (x, w))
    output = runtime(x, w)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Output sum: {output.sum():.4f}")

    print("\n--- Second call (should use cached graph) ---")
    output2 = runtime(torch.randn(4, 8), w)
    print(f"Output2 sum: {output2.sum():.4f}")


if __name__ == "__main__":
    main()
