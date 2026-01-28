"""Pipeline demo: partition → compile → serialize → runtime."""
import json
from typing import List

import torch
from torch.fx import GraphModule

import kernel_lib_c_abi as c_abi


OP_TO_KERNEL = {
    "aten.matmul.default": "gemm",
    "aten.relu.default": "relu",
    "aten.softmax.int": "softmax",
    "aten.softmax.default": "softmax",
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


def partition_graph(gm: GraphModule):
    """Return accelerator vs unsupported segments from a linear scan."""
    segments = []
    current_kind = None
    current_nodes = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        is_supported = _target_to_str(node.target) in OP_TO_KERNEL
        kind = "accelerator" if is_supported else "unsupported"
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
    """Lower an FX graph into a linear program with support tags."""
    program = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            program.append({"op": "placeholder", "name": node.name})
        elif node.op == "call_function":
            target_str = _target_to_str(node.target)
            kernel = OP_TO_KERNEL.get(target_str)
            program.append(
                {
                    "op": "call_function",
                    "name": node.name,
                    "target": target_str,
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
    """Serialize a program to bytes (mock compiler artifact)."""
    return json.dumps(program, default=str).encode("utf-8")


def deserialize_program(blob):
    return json.loads(blob.decode("utf-8"))


def my_accel_backend(gm: GraphModule, example_inputs: List[torch.Tensor]):
    """Compile + serialize a graph and return a runtime callable."""
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

                if not instr["supported"]:
                    raise NotImplementedError(
                        f"Unsupported op '{instr['target']}' in accelerator-only mode"
                    )
                result = c_abi.call_kernel(instr["kernel"], *fn_args, **fn_kwargs)

                env[instr["name"]] = result
            elif instr["op"] == "output":
                names = instr["names"]
                if len(names) == 1:
                    return env[names[0]]
                return tuple(env[name] for name in names)

        return None

    return runtime_executor


class DemoModel(torch.nn.Module):
    def forward(self, x, w):
        h = torch.relu(x)
        y = torch.matmul(h, w)
        z = torch.softmax(y, dim=-1)
        return z


def main():
    print("Testing pipeline backend demo\n")

    x = torch.randn(4, 8)
    w = torch.randn(8, 16)

    model = DemoModel()
    exported = torch.export.export(model, (x, w))
    gm = exported.graph_module
    runtime = my_accel_backend(gm, (x, w))
    output = runtime(x, w)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Output sum: {output.sum():.4f}")


if __name__ == "__main__":
    main()
