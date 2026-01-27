"""Export examples for graph capture and inspection."""
import torch
class SimpleModule(torch.nn.Module):
    def forward(self, x, y):
        return torch.relu(x @ y.T)


def _unwrap_output(output):
    """Return the tensor when export produces a single-item tuple."""
    if isinstance(output, tuple) and len(output) == 1:
        return output[0]
    return output


def run_example_1():
    """Capture and run a simple matmul+relu graph."""
    x = torch.randn(128, 64)
    y = torch.randn(128, 64)
    model = SimpleModule()
    exported = torch.export.export(model, (x, y))
    result = _unwrap_output(exported.graph_module(x, y))
    print(f"Example 1 - Output shape: {result.shape}")


def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    """Print a graph summary and return the graph callable."""
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


class TraceModule(torch.nn.Module):
    def forward(self, x, w):
        h = torch.relu(x)
        y = torch.matmul(h, w)
        z = torch.softmax(y, dim=-1)
        return z


def run_example_2():
    """Export a graph and print it through the custom backend."""
    print("\n\nExample 2 - Custom Backend:")
    x = torch.randn(32, 128)
    w = torch.randn(128, 64)
    model = TraceModule()
    exported = torch.export.export(model, (x, w))
    output = _unwrap_output(my_custom_backend(exported.graph_module, (x, w))(x, w))
    print(f"Output shape: {output.shape}")


class ExportOpsModule(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x) + torch.cos(x)


def run_example_3():
    """Show exported ops and run the graph module."""
    print("\n\nExample 3 - Exported ops:")
    x = torch.randn(100)
    model = ExportOpsModule()
    exported = torch.export.export(model, (x,))
    exported.graph_module.graph.print_tabular()
    result = _unwrap_output(exported.graph_module(x))
    print(f"Example 3 - Output sum: {result.sum():.4f}")


if __name__ == "__main__":
    run_example_1()
    run_example_2()
    run_example_3()
