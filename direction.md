# Atalla PyTorch Infra — Direction (Condensed)

## Recommendation

Use **torch.export + ExecuTorch‑style delegate** as the main pipeline, and run
**PrivateUse1** in parallel for kernel bring‑up and allocator validation. The
target artifact is a **final C executable** (no eager execution).

## Why

- Export/delegate gives a clean **graph → blob → runtime** story.
- CPU fallback is not used; all ops must be lowered to Atalla.
- PrivateUse1 accelerates kernel verification without blocking compiler work.
- Static export avoids JIT/firmware complexity and supports DRAM‑loaded blobs.

## Options (brief)

- **ExecuTorch delegate**: best AOT story, Python AOT + C++ runtime.
- **PrivateUse1**: strong for kernel + memory plumbing, not full pipeline alone.
- **ONNX/TVM/MLIR**: valuable later, heavier upfront cost.

## Near‑Term Operator Set

- Core: `aten.matmul.default`, `aten.relu.default`, `aten.softmax.int`
- Elementwise: `aten.add.Tensor`, `aten.sub.Tensor`, `aten.mul.Tensor`

## Phased Plan

- **Phase 1**: kernel stubs + blob format + runtime loop.
- **Phase 2**: memory planner, layouts/strides, shape specialization.
- **Phase 3**: simulator/RTL integration + profiling + model suite.

## Risks / Watch‑outs

- Memory layout mismatches and alignment bugs.
- Under‑specifying the blob format early.
- Dynamic shape support too early (avoid for initial demo).
