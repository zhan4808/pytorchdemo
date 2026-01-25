# Deep Analysis: torch.compile + FX + Custom Backend

This document explains what each script does, how the implementation maps to
the `torch.compile` flow, and how to read the output conceptually.

## Overall Flow (Conceptual)

1. User writes eager PyTorch code.
2. `torch.compile` uses TorchDynamo to intercept Python frames and build an FX
   graph.
3. The backend receives a `GraphModule` (FX IR + generated Python).
4. The backend returns a callable that executes the graph (or compiled code).

Key idea: the backend sees a graph of ops (not raw Python), and can inspect,
transform, or lower those ops to another target.

---

## 1) `test_compile_basics.py`

### What the script does

This file has three independent demos:

1. **Example 1**: A basic `torch.compile` function using matrix multiply and
   ReLU.
2. **Example 2**: A **custom backend** that prints the FX graph and returns
   `gm.forward` for execution.
3. **Example 3**: A compiled function that triggers Inductor logs to show the
   generated code paths.

### Implementation details and conceptual flow

#### Example 1
```python
@torch.compile
def simple_fn(x, y):
    return torch.relu(x @ y.T)
```
- Dynamo intercepts the Python bytecode when `simple_fn` is first called.
- It builds an FX graph with `matmul` and `relu`.
- Because no custom backend is specified, it uses the default backend
  (Inductor).
- The output is just the result of running the compiled graph.

**Why the output shape is `torch.Size([128, 128])`**  
`x` is `[128, 64]` and `y.T` is `[64, 128]`, so the matmul result is
`[128, 128]`. ReLU does not change shape.

#### Example 2
```python
def my_custom_backend(gm: torch.fx.GraphModule, example_inputs):
    ...
    gm.graph.print_tabular()
    print(gm.code)
    return gm.forward
```
- Dynamo builds an FX graph for `model_to_trace`.
- The backend receives the graph and prints:
  - Node list (`placeholder`, `call_function`, `output`)
  - Tabular graph view
  - Generated Python code for the graph
- The backend returns `gm.forward`, so execution uses the graph exactly as
  printed.

**Why the graph looks like it does**  
The input program is:
```python
def model_to_trace(x, w):
    h = torch.relu(x)
    y = torch.matmul(h, w)
    z = torch.softmax(y, dim=-1)
    return z
```

So the FX graph contains:
- 2 placeholders (`x`, `w`)
- 3 `call_function` nodes (`relu`, `matmul`, `softmax`)
- 1 output node

The generated code in the output is a faithful, lowered version of those ops.

#### Example 3
```python
os.environ["TORCH_LOGS"] = "output_code"

@torch.compile
def inductor_example(x):
    return torch.sin(x) + torch.cos(x)
```
- `TORCH_LOGS=output_code` asks Inductor to log generated code and decisions.
- `torch.compile` runs the function through Dynamo + Inductor.
- You should see the output sum, plus additional Inductor logging in the
  console depending on your PyTorch version.

**Why the sum prints**  
The function returns a tensor; we print `result.sum()` to confirm the compiled
path executed successfully.

### How to read your output

Your output shows:
- Example 1: the expected `[128, 128]` shape.
- Example 2: the graph dump, tabular graph, and code, followed by a `[32, 64]`
  output shape.
- Example 3: a message + a numeric sum.

Conceptually, this proves:
- Dynamo is capturing the graph.
- The backend receives the graph and can inspect it.
- The compiled graph executes correctly.

---

## 2) `explore_fx_graph.py`

### What the script does

This script uses `torch.fx.symbolic_trace` to create an FX graph for a small
`nn.Module`. It prints:
- The FX graph table
- Each node’s attributes
- The generated Python code

### Implementation details and conceptual flow

```python
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)

    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x
```
- `symbolic_trace` runs a tracer that records ops instead of executing them
  normally.
- `call_module` appears for `self.linear` because it is a submodule.
- `call_function` appears for `torch.relu`.

**Why the graph shows `call_module` and `call_function`**
- `self.linear` is a module attribute, so FX models it as a `call_module`.
- `torch.relu` is a functional call, so it appears as a `call_function`.

### How to read your output

The output confirms:
- An input placeholder for `x`
- A `call_module` node for `linear`
- A `call_function` node for `relu`
- An output node

This is the base IR that Dynamo uses after tracing, so understanding this is
foundational for backend work.

---

## 3) `my_accel_backend_prototype.py`

### What the script does

This file simulates a custom backend:
- It defines a tiny **op registry** for `matmul`, `relu`, and `softmax`.
- It inspects the FX graph and reports supported ops.
- It runs a **custom executor** that dispatches each graph op through the
  registry (or falls back to PyTorch).

### Implementation details and conceptual flow

#### Kernel library
```python
import kernel_lib as kernels
```
- `kernel_lib.py` holds placeholder kernel implementations in Python.
- These functions are the stand-ins for real C/C++/ISA kernels.

#### Op registry
```python
OP_REGISTRY = {
    torch.matmul: my_accel_matmul,
    torch.relu: my_accel_relu,
    torch.softmax: my_accel_softmax,
    ...
}
```
- This models the "lowering" phase to a device-specific kernel library.
- In real backends, this would route into a C/C++/ISA implementation.

#### Backend function
```python
def my_accel_backend(gm: GraphModule, example_inputs):
    ...
    return custom_executor
```
- Dynamo calls this once per compiled graph.
- We inspect and summarize supported ops, then return a callable.

#### Compiler stage (lowering to a program)
```python
def compile_graph(gm: GraphModule):
    ...
    return program
```
- This models a compiler step that turns the FX graph into a linear program.
- The program stores op targets, args, and whether each op is supported.
- This is the separation between **compile time** and **runtime**.

#### Custom executor
```python
def custom_executor(*args):
    ...
    for instr in program:
        if instr["op"] == "call_function":
            # use registry or fallback
```
- This is a minimal "interpreter" for the FX graph.
- It manually executes the graph in topological order.
- Each op is dispatched to the custom kernel (or fallback).

### How to read your output

Your output shows:
- The backend received the graph.
- All ops were recognized as supported.
- The graph was compiled into a program before execution.
- Each op execution prints from the custom kernel.
- Two runs show the cached graph path on the second call.

Conceptually, this proves:
- You can intercept the FX graph and build your own executor.
- You can route ops to custom implementations.
- You understand fallback for unsupported ops.

### Output deep dive (your run)

```text
MY_ACCEL BACKEND - GRAPH RECEIVED
  Found op: relu - ✓ SUPPORTED
  Found op: matmul - ✓ SUPPORTED
  Found op: softmax - ✓ SUPPORTED

Compiling graph -> 3 call_function nodes
Program length: 5
```
- The backend sees the FX graph and enumerates each `call_function` node.
- Each op matches the registry, so no fallbacks are needed.
- The compiler emits a short linear program (placeholders + call_function ops + output).

```text
--- EXECUTING ON MY_ACCEL ---
  [MY_ACCEL] relu called: torch.Size([4, 8])
  [MY_ACCEL] matmul called: torch.Size([4, 8]) @ torch.Size([8, 16])
  [MY_ACCEL] softmax called: torch.Size([4, 16]), dim=-1
```
- These prints come from the per-op wrappers that call `kernel_lib`.
- `relu` keeps shape `[4, 8]`.
- `matmul` produces `[4, 16]`.
- `softmax` is applied across the last dimension.

```text
Final output shape: torch.Size([4, 16])
Output sum: 4.0000
```
- Softmax makes each row sum to 1, so the total sum is ~4.0 for 4 rows.

```text
--- Second call (should use cached graph) ---
```
- Dynamo reuses the compiled graph; the backend is not reinvoked.
- The executor runs again with new inputs using the same graph.

---

## Are you done with the initial requirements?

Yes. The initial requirements were to:
- Stand up a working PyTorch environment with `torch.compile`
- Demonstrate the compile flow and graph capture
- Write a custom backend that inspects the FX graph
- Run small experiments and interpret the output

Your runs show all of these working.

---

## What to run and what to see (quick checklist)

1. `python test_compile_basics.py`
   - Graph capture + backend inspection + compiled output
2. `python explore_fx_graph.py`
   - FX graph structure and node types
3. `python my_accel_backend_prototype.py`
   - Backend registry + custom executor + cached graph reuse

---

## 4) `backend_pipeline_demo.py`

### What the script does

This file models a more realistic pipeline:
- **Partitioning**: splits supported vs fallback ops in a linear pass.
- **Compilation**: lowers FX into a compact linear "program".
- **Serialization**: emits a byte blob (compiler artifact).
- **Runtime**: loads the blob and executes it via a simulated C ABI.

### Implementation details and conceptual flow

#### Partitioning stage
```python
segments = partition_graph(gm)
```
- The graph is scanned for `call_function` nodes.
- Supported ops are grouped into accelerator segments.
- Unsupported ops go to fallback segments (CPU).

This mirrors ExecuTorch/PrivateUse1 style partitioning.

#### Compilation stage
```python
program = compile_graph(gm)
```
- Each FX node becomes a compact instruction with args/kwargs encoded.
- Each op is tagged with its kernel name or fallback flag.

#### Serialization stage
```python
blob = serialize_program(program)
runtime_program = deserialize_program(blob)
```
- Models the "compiler artifact" your real toolchain would emit.
- The runtime reads a blob and executes without access to the FX graph.

#### Runtime stage
```python
result = c_abi.call_kernel(instr["kernel"], *fn_args, **fn_kwargs)
```
- Crosses a simulated C ABI boundary (`kernel_lib_c_abi.py`).
- This mirrors how Python would call into a real kernel library.

### How to read the output

Typical output:
```text
PIPELINE BACKEND - GRAPH RECEIVED
Partitioned into 1 segment(s)
  Segment 0: accelerator (3 ops)
Compiled program length: 5
Serialized program bytes: <n>

--- RUNTIME EXECUTION ---
```

Interpretation:
- All ops were supported, so only accelerator segments were created.
- Program length includes placeholders + ops + output.
- Serialization confirms a backend artifact boundary.
- Runtime executes through the C ABI layer.
