# Atalla PyTorch Infra — Report

## Executive Summary

We should build the Atalla PyTorch infrastructure around **`torch.export` and an
ExecuTorch‑style delegate** that emits a **static blob** loaded into simulated
DRAM, then executed by the runtime through a C‑ABI kernel library. This avoids
JIT/firmware complexity while matching how hardware and compiler teams want to
reason about fixed programs and memory layouts. In parallel, a minimal
**PrivateUse1** backend should be used for kernel bring‑up and allocator
validation, without blocking the main pipeline.

## Context and Constraints

- **Goal**: run real models on Atalla (including LLM‑style workloads) with a
  software pipeline that is credible for simulation/RTL and future hardware.
- **Constraint from lead**: avoid `torch.compile`/JIT because it requires a
  firmware library for tile transfers and heterogeneous communication, plus a
  simulated CPU to orchestrate runtime JIT execution. This adds major complexity
  and breaks clean benchmarking.
- **Implication**: we need **static graph capture** and **fixed executables**
  loaded into simulated DRAM.

## Proposed Architecture (Atalla‑Optimized)

1. **Graph capture**: `torch.export.export` produces an FX `GraphModule` with
   ATen ops (e.g., `aten.matmul.default`).
2. **Partitioning**: split supported vs fallback ops. Supported subgraphs are
   delegated; fallback executes on CPU.
3. **Lowering**: map ATen ops to Atalla kernel stubs (op registry).
4. **Compilation**: lower to a linear program IR or Atalla‑specific IR.
5. **Serialization**: emit a versioned blob, placed into simulated DRAM.
6. **Runtime**: read blob, launch Atalla kernels through the C‑ABI boundary.
7. **Simulator/RTL hooks**: runtime dispatches to simulator when enabled.

This mirrors hardware expectations: a **static program** in memory, deterministic
execution, and explicit control over data movement and layouts.

## Why `torch.export` + Delegate is the Best Fit

- **Static program**: aligns with simulated DRAM + fixed executables.
- **Cleaner debugging**: FX graph is inspectable and deterministic.
- **CPU fallback**: partial coverage works early with minimal ops.
- **Separation of concerns**: compiler handles blob; runtime executes blob.
- **Matches existing projects**: ExecuTorch delegates are explicitly designed
  for ahead‑of‑time compilation and embedded deployment.

## Alternatives Considered

### 1) PrivateUse1 Device Backend
**Pros**
- Direct kernel validation and allocator wiring in PyTorch.
- Useful for early kernel bring‑up and unit testing.

**Cons**
- Not a full graph → blob → runtime story.
- Harder to present as end‑to‑end deployment pipeline.

**Conclusion**: Keep as a **parallel track**, not the main pipeline.

### 2) ONNX Runtime Execution Provider (EP)
**Pros**
- Mature infrastructure and multi‑framework support.
- Good fallback semantics.

**Cons**
- In‑tree build requirement, heavier integration.
- Less PyTorch‑native than export + delegate.

**Conclusion**: Viable later, but not optimal for near‑term Atalla demo.

### 3) TVM BYOC
**Pros**
- Strong compiler stack and auto‑tuning capabilities.

**Cons**
- Steep learning curve and heavier engineering cost.
- Adds framework conversion overhead.

**Conclusion**: Possible future direction, not the fastest path for a demo.

### 4) MLIR / torch‑mlir
**Pros**
- Deep compiler flexibility; strong for research.

**Cons**
- Highest integration cost; more moving parts.

**Conclusion**: Too heavy for current goals; revisit when compiler stack matures.

## Existing Reference Projects and Lessons

### ExecuTorch Delegates
- AOT partitioner + preprocess compile step on Python side.
- C++ runtime with `init()` and `execute()` is a good model for simulator hooks.

### Gemmini‑style Stacks
- Parameterized C kernel library tied to hardware configuration.
- Static loadables and fixed memory layouts are key for systolic arrays.

### NVDLA‑style Loadables
- Compiler emits a loadable binary; runtime consumes it directly.
- This is the closest conceptual match to Atalla’s desired stack.

**Takeaway**: successful stacks define a **stable binary format** and strict
kernel contracts early.

## Operator Set for Initial Demo

**Core ops**
- `aten.matmul.default`
- `aten.relu.default`
- `aten.softmax.int`

**Elementwise ops**
- `aten.add.Tensor`
- `aten.sub.Tensor`
- `aten.mul.Tensor`

This is enough to validate GEMM‑centric kernels and basic transformer blocks.

## Key Components We Still Need

- **Kernel library**: real C/C++ implementations for GEMM/softmax/relu.
- **Memory planner**: explicit buffers, layouts, strides, alignment rules.
- **Shape specialization**: fixed shapes initially; guard strategy later.
- **Binary format**: versioned blob with metadata (layouts, constants).
- **Simulator hooks**: runtime dispatch into RTL/sim with debug telemetry.

## Scratchpad Allocation Model (from HW/Compiler team)

Atalla scratchpad is **not** allocated during lowering, and it is **not**
owned by each kernel. Instead, kernels call a **scratchpad allocation
library function** at runtime:

- The allocator returns a free page address.
- Kernels use that address for temporary buffers.
- This implies a firmware‑style support layer that manages allocation,
  bookkeeping, and (eventually) reuse policies.

**Implication for the pipeline:** the runtime must expose these allocator
entry points and the kernel library must call them. The compiler only
needs to respect scratchpad constraints (page size, alignment, bank layout)
in its code generation assumptions.

## Kernel vs Firmware Responsibilities

- **Kernels**: app‑specific compute (GEMM, softmax, etc.).
- **Firmware/library calls**: system services like scratchpad allocation,
  DMA/transfer orchestration, synchronization, and bookkeeping.

This split mirrors the conversation: “Imagine library calls… kernel is
app‑specific… these lib calls are functions handling system stuff like a
firmware.” It keeps kernels small and lets system policy evolve independently.

## Risks and Watch‑Outs

- **Memory layout mismatches**: most common correctness bugs.
- **Under‑specified blob format**: blocks compiler/runtime alignment later.
- **Dynamic shapes too early**: avoid until basic correctness is stable.
- **Debug infrastructure**: needs to be in place early for RTL bring‑up.

## Phased Plan

**Phase 1 (now)**
- Kernel stubs + blob format + runtime loop + CPU fallback.
- Minimal operator set, correctness first.

**Phase 2**
- Memory planner, layout/stride handling, shape specialization.

**Phase 3**
- Simulator/RTL integration, profiling, model suite.

## Why This is Ideal for Atalla

- **Systolic arrays** benefit from static scheduling and explicit layouts.
- **Compiler‑friendly**: clear interface for kernel teams and compiler teams.
- **Simulation‑friendly**: deterministic blobs and DRAM placement.
- **Scalable**: incremental operator coverage with fallback.

## Ask for Alignment

- Confirm target operator set and kernel signatures.
- Agree on blob format/versioning and metadata.
- Define simulator ABI and logging/trace requirements.
