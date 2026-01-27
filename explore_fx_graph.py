"""FX tracing demo for inspecting graph structure."""
import torch
from torch.fx import symbolic_trace


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x


def main():
    """Trace a module and print graph details."""
    model = SimpleModel()
    traced = symbolic_trace(model)

    print("=" * 60)
    print("FX GRAPH EXPLORATION")
    print("=" * 60)

    print("\n1. Graph structure:")
    traced.graph.print_tabular()

    print("\n2. Walking through nodes:")
    for node in traced.graph.nodes:
        print(
            f"""
    Node: {node.name}
    - op: {node.op}
    - target: {node.target}
    - args: {node.args}
    - kwargs: {node.kwargs}
    """
        )

    print(
        """
3. Node types in FX graphs:
   - placeholder: Input to the graph
   - call_function: Calls a function (e.g., torch.relu)
   - call_method: Calls a tensor method (e.g., x.view())
   - call_module: Calls a submodule (e.g., self.linear)
   - get_attr: Gets an attribute
   - output: The return value
"""
    )

    print("4. Generated forward code:")
    print(traced.code)


if __name__ == "__main__":
    main()
