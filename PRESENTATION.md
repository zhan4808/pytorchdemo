# Atalla PyTorch Infra Proposal (10 min)

## TL;DR

Adopt **`torch.export` + ExecuTorch‑style delegate** as the main Atalla pipeline.
Run **PrivateUse1** in parallel for kernel bring‑up and allocator validation.
This gives a clean **graph → blob → runtime** story with CPU fallback.

## Decision Summary

- **Primary path**: `torch.export` → partition → compile → blob → runtime (C‑ABI)
- **Parallel path**: PrivateUse1 for kernel validation and memory plumbing
- **Why**: best alignment with simulator/RTL and model‑execution goals

## Options (10‑second comparison)

- **ExecuTorch delegate**: fastest end‑to‑end AOT story + fallback
- **PrivateUse1**: great for kernels, incomplete as a full pipeline
- **ONNX/TVM**: valuable later, too heavy for near‑term demo

## Proposed Atalla Stack

1. **Graph capture**: `torch.export.export` → FX `GraphModule`
2. **Lowering**: ATen ops → Atalla op registry
3. **Compilation**: FX → linear program (IR)
4. **Serialization**: program → binary blob (versioned)
5. **Runtime**: blob → kernel calls (C‑ABI)
6. **Fallback**: unsupported ops → `torch.ops.*` (CPU)
7. **Simulator hook**: runtime dispatch when `ATALLA_SIM=1`

## Near‑Term Operator Set

- **Core**: `aten.matmul.default`, `aten.relu.default`, `aten.softmax.int`
- **Elementwise**: `aten.add.Tensor`, `aten.sub.Tensor`, `aten.mul.Tensor`

## 10‑Minute Talk Track

1. **Goal**: run real models on Atalla with a credible software pipeline
2. **Current demo**: export → compile → blob → runtime → kernel stubs
3. **Decision**: ExecuTorch‑style delegate is the spine; PrivateUse1 is parallel
4. **Architecture**: show pipeline diagram
5. **Roadmap**: kernels → memory → blob → simulator → model suite
6. **Asks**: op set, blob format, simulator ABI

## Phase Plan

- **Phase 1 (now)**: GEMM/softmax/relu + add/sub/mul, CPU fallback, demo models
- **Phase 2**: memory planner, layouts/strides, shape specialization
- **Phase 3**: stable blob format, simulator/RTL hooks, profiling

## Asks (Team Alignment)

- Confirm target op set for demo models
- Agree on blob format versioning
- Define simulator ABI and expected latency model
