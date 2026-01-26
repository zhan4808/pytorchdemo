# Atalla PyTorch Infra — Direction (Condensed)

## Recommendation

Use **torch.export + ExecuTorch‑style delegate** as the main pipeline, and run
**PrivateUse1** in parallel for kernel bring‑up and allocator validation.

## Why

- Export/delegate gives a clean **graph → blob → runtime** story.
- CPU fallback enables partial coverage early.
- PrivateUse1 accelerates kernel verification without blocking compiler work.

## Options (brief)

- **ExecuTorch delegate**: best AOT story + fallback, Python AOT + C++ runtime.
- **PrivateUse1**: strong for kernel + memory plumbing, not full pipeline alone.
- **ONNX/TVM/MLIR**: valuable later, heavier upfront cost.

## Near‑Term Operator Set

- Core: `aten.matmul.default`, `aten.relu.default`, `aten.softmax.int`
- Elementwise: `aten.add.Tensor`, `aten.sub.Tensor`, `aten.mul.Tensor`

## Phased Plan

- **Phase 1**: kernel stubs + blob format + runtime loop + CPU fallback.
- **Phase 2**: memory planner, layouts/strides, shape specialization.
- **Phase 3**: simulator/RTL integration + profiling + model suite.

## Risks / Watch‑outs

- Memory layout mismatches and alignment bugs.
- Under‑specifying the blob format early.
- Dynamic shape support too early (avoid for initial demo).
