import os
import torch


# ============================================
# Example 1: Basic torch.compile
# ============================================
@torch.compile
def simple_fn(x, y):
    return torch.relu(x @ y.T)


def run_example_1():
    x = torch.randn(128, 64)
    y = torch.randn(128, 64)
    result = simple_fn(x, y)
    print(f"Example 1 - Output shape: {result.shape}")


# ============================================
# Example 2: Custom backend that prints the graph
# ============================================
def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    """
    Custom backend that prints graph info and returns a callable.
    """
    print("\n" + "=" * 60)
    print("MY CUSTOM BACKEND RECEIVED A GRAPH!")
    print("=" * 60)

    print("\nGraph nodes:")
    for node in gm.graph.nodes:
        print(f"  {node.op:15} | {node.name:15} | {node.target}")

    print("\nTabular format:")
    gm.graph.print_tabular()

    print("\nGenerated Python code:")
    print(gm.code)

    return gm.forward


@torch.compile(backend=my_custom_backend)
def model_to_trace(x, w):
    h = torch.relu(x)
    y = torch.matmul(h, w)
    z = torch.softmax(y, dim=-1)
    return z


def run_example_2():
    print("\n\nExample 2 - Custom Backend:")
    x = torch.randn(32, 128)
    w = torch.randn(128, 64)
    output = model_to_trace(x, w)
    print(f"Output shape: {output.shape}")


# ============================================
# Example 3: See what ops Inductor receives
# ============================================
@torch.compile
def inductor_example(x):
    return torch.sin(x) + torch.cos(x)


def run_example_3():
    os.environ["TORCH_LOGS"] = "output_code"
    print("\n\nExample 3 - Inductor output (check console for generated code):")
    result = inductor_example(torch.randn(100))
    print(f"Example 3 - Output sum: {result.sum():.4f}")


if __name__ == "__main__":
    run_example_1()
    run_example_2()
    run_example_3()
