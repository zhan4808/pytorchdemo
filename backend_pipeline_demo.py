"""
Demo: partition + compile + serialize + execute pipeline.
This models an ExecuTorch/PrivateUse1-style flow with CPU fallback.
"""
import json
from typing import List

import torch
from torch.fx import GraphModule

import kernel_lib_c_abi as c_abi


OP_TO_KERNEL = {
    torch.matmul: "gemm",
    torch.relu: "relu",
    torch.nn.functional.relu: "relu",
    torch.softmax: "softmax",
    torch.nn.functional.softmax: "softmax",
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


def partition_graph(gm: GraphModule):
    """
    Partition into supported and fallback ops in a single linear pass.
    Returns a list of segments: [{"kind": "accelerator"|"fallback", "nodes": [...]}].
    """
    segments = []
    current_kind = None
    current_nodes = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        is_supported = node.target in OP_TO_KERNEL
        kind = "accelerator" if is_supported else "fallback"
        if current_kind is None:
            current_kind = kind
        if kind != current_kind:
            segments.append({"kind": current_kind, "nodes": current_nodes})
            current_nodes = []
            current_kind = kind
        current_nodes.append(node)

    if current_nodes:
        segments.append({"kind": current_kind, "nodes": current_nodes})

    return segments


def compile_graph(gm: GraphModule):
    """
    Lower the FX graph to a linear program with an explicit fallback flag.
    """
    program = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            program.append({"op": "placeholder", "name": node.name})
        elif node.op == "call_function":
            kernel = OP_TO_KERNEL.get(node.target)
            program.append(
                {
                    "op": "call_function",
                    "name": node.name,
                    "target": node.target,
                    "kernel": kernel,
                    "supported": kernel is not None,
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
    return program


def serialize_program(program):
    """
    Serialize to bytes (mock compiler artifact).
    """
    return json.dumps(program, default=str).encode("utf-8")


def deserialize_program(blob):
    return json.loads(blob.decode("utf-8"))


def my_accel_backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    print("\n" + "=" * 60)
    print("PIPELINE BACKEND - GRAPH RECEIVED")
    print("=" * 60)

    segments = partition_graph(gm)
    print(f"Partitioned into {len(segments)} segment(s)")
    for idx, seg in enumerate(segments):
        print(f"  Segment {idx}: {seg['kind']} ({len(seg['nodes'])} ops)")

    program = compile_graph(gm)
    print(f"Compiled program length: {len(program)}")

    blob = serialize_program(program)
    print(f"Serialized program bytes: {len(blob)}")

    runtime_program = deserialize_program(blob)

    def runtime_executor(*args):
        print("\n--- RUNTIME EXECUTION ---")

        env = {}
        arg_iter = iter(args)

        for instr in runtime_program:
            if instr["op"] == "placeholder":
                env[instr["name"]] = next(arg_iter)
            elif instr["op"] == "call_function":
                fn_args = [_decode_arg(a, env) for a in instr["args"]]
                fn_kwargs = {k: _decode_arg(v, env) for k, v in instr["kwargs"].items()}

                if instr["supported"]:
                    result = c_abi.call_kernel(instr["kernel"], *fn_args, **fn_kwargs)
                else:
                    print(f"  [FALLBACK] {instr['target']}")
                    result = instr["target"](*fn_args, **fn_kwargs)

                env[instr["name"]] = result
            elif instr["op"] == "output":
                names = instr["names"]
                if len(names) == 1:
                    return env[names[0]]
                return tuple(env[name] for name in names)

        return None

    return runtime_executor


@torch.compile(backend=my_accel_backend)
def demo_model(x, w):
    h = torch.relu(x)
    y = torch.matmul(h, w)
    z = torch.softmax(y, dim=-1)
    return z


def main():
    print("Testing pipeline backend demo\n")

    x = torch.randn(4, 8)
    w = torch.randn(8, 16)

    output = demo_model(x, w)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Output sum: {output.sum():.4f}")


if __name__ == "__main__":
    main()
