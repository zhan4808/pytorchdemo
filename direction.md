# Getting PyTorch models onto custom accelerators: a practical guide for research projects

For a systolic array accelerator at the RTL simulation stage with **10 months to demo**, the fastest path to PyTorch integration is adopting Gemmini's ONNX-based approach combined with ExecuTorch delegates for the framework layer. This combination provides the best balance between implementation effort and production readiness. The critical insight from successful academic projects is that **full-stack integration matters more than any single component**—Gemmini's success came from treating software as co-equal with hardware from day one.

The research reveals a clear hierarchy: ONNX Runtime or ExecuTorch delegates require the least infrastructure (~2-3 months to basic integration), TVM BYOC offers powerful optimization but steeper learning curves (~3-6 months), and MLIR-based flows provide maximum flexibility at highest implementation cost. Given your current state—working RTL simulation, defined ISA, basic assembly kernels—the pragmatic path starts with assembly kernel validation on simulation, builds a minimal code generator targeting those kernels, and integrates with PyTorch via the **PrivateUse1** mechanism or ExecuTorch delegates with CPU fallback for unsupported operations.

## Gemmini's three-level software stack offers a proven template

Berkeley's Gemmini accelerator represents the gold standard for academic accelerator software stacks, having been fabricated in **TSMC 16nm** and **Intel 22nm** with complete framework integration. Their architecture uses three programming levels that serve different users.

At the highest level, Gemmini uses **onnxruntime-riscv**, a fork of Microsoft's ONNX Runtime, providing push-button workflow from PyTorch models. The integration path flows: PyTorch → ONNX export → onnxruntime-riscv → Systolic Execution Provider → Gemmini C library → custom RISC-V instructions → hardware. This ONNX-based approach was deliberately chosen because it provides an industry-standard intermediate representation with existing tooling.

The middle level consists of a hand-tuned C library (`gemmini.h`) wrapping custom instructions into standard DNN operators. A critical innovation is that this library is **auto-parameterized**—the hardware generator produces a `gemmini_params.h` header containing systolic array dimensions, supported dataflows, and scratchpad sizes. Every new hardware instantiation automatically generates matching software parameters, eliminating manual synchronization between hardware and software configurations.

At the lowest level, C macros construct RISC-V instruction encodings since GNU binutils doesn't expose custom instructions. Gemmini's CISC-style loop instructions (`gemmini_loop_ws`, `gemmini_loop_conv_ws`) implement automatic tiling and double-buffering through FSM-based monitoring of ROB occupancy—this hardware-software co-design maximizes load/store/execute overlap without complex compiler scheduling.

The key lesson from Gemmini is that **running Linux exposed bugs that baremetal testing never found**—non-deterministic deadlocks during context switches, memory permission issues, and TLB miss rates reaching 20-30% on DNN workloads. Their recommendation: invest in full SoC integration early rather than treating the accelerator in isolation.

## TVM's BYOC mechanism enables framework-agnostic integration

Apache TVM's BYOC (Bring Your Own Codegen) provides the most flexible approach for integrating custom accelerators while maintaining compatibility with PyTorch, TensorFlow, and ONNX through unified Relay IR. The mechanism works through four steps: annotation (declaring supported operators), graph transformation (partitioning the model), code generation (producing executable code), and runtime execution.

A minimal BYOC implementation requires **three components totaling ~900-1,500 lines of code**: Python annotation rules declaring which operators your accelerator supports (~100 LOC), C++ codegen generating calls to your kernel library (~500 LOC), and a runtime wrapper handling execution (~300 LOC). The 2024 Gemmini study showed that automated tooling can reduce this to approximately **200 lines of YAML and Python**—an 80% reduction in manual effort.

The integration path from PyTorch proceeds: `torch.jit.trace(model)` → `relay.frontend.from_pytorch()` → `MergeComposite` (fuse patterns) → `AnnotateTarget` (mark supported regions) → `PartitionGraph` → compile for your backend. TVM handles heterogeneous execution automatically—unsupported operators fall back to CPU.

TVM's VTA (Versatile Tensor Accelerator) serves as an excellent reference implementation, providing a two-level ISA (CISC task-level and RISC micro-op level) deployable to FPGA (~$200 Pynq board) or behavioral simulator. The JIT runtime compiles accelerator binaries on-the-fly, enabling rapid iteration during development.

The primary disadvantages are a **steep learning curve** for TVM internals (Relay, TIR, pass infrastructure) and documentation gaps in some areas. UMA (Universal Modular Accelerator), TVM's simplified Python-based interface, lacks scheduling and tiling support needed for complex accelerators. Budget **2-4 weeks** for basic BYOC integration, **1-2 months** for JSON runtime with graph engine, and **3-6 months** for full VTA-style integration with auto-tuning.

## ExecuTorch delegates offer the most direct PyTorch path

ExecuTorch, PyTorch's edge deployment framework, provides the most native integration through its delegate system—subgraphs are compiled and dispatched to custom backends while maintaining direct compatibility with `torch.export()`. The architecture splits into ahead-of-time (Python) and runtime (C++) components.

The AOT interface requires implementing a **Partitioner** (algorithm tagging nodes for delegation) and **Preprocess** function (compiling tagged subgraphs to binary blobs). The runtime interface needs `init()` (parse blob, setup accelerator), `execute()` (run delegated computation), and optionally `destroy()` (cleanup). A minimal implementation requires approximately **200 lines of Python** for AOT compilation and **150 lines of C++** for runtime execution.

Existing delegate examples provide excellent templates: XNNPACK (reference CPU implementation), Qualcomm QNN (NPU with quantization), Apple CoreML and MPS, and ARM Ethos-U (which demonstrates **Fixed Virtual Platform/simulator support**—directly relevant for RTL simulation scenarios). The ARM Ethos-U delegate shows how `execute()` can call a simulator before hardware exists.

The key advantages for early-stage accelerators include incremental operator support with automatic CPU fallback, Python-first development for rapid iteration, and compiled blobs that can be **any format**—your ISA, high-level IR, or serialized graph. The ~50KB base runtime footprint makes ExecuTorch suitable for embedded deployment.

The disadvantages include requiring both Python and C++ components, CMake integration, and less mature documentation for academic use cases compared to TVM. The recommended starting point is the `BackendWithCompilerDemo` reference implementation showing a complete minimal delegate.

## ONNX Runtime execution providers work best with existing kernel libraries

ONNX Runtime's Execution Provider (EP) framework enables custom hardware integration through a well-defined C++ interface. The core mechanism: EPs declare capabilities via `GetCapability()`, claimed subgraphs are compiled via `Compile()`, and execution dispatches to compiled kernels with automatic CPU fallback.

The minimum implementation requires implementing `IExecutionProvider` with `GetCapability()` (return supported nodes/subgraphs), `Compile()` (generate fused kernels), `GetAllocator()` (provide memory allocator), and `Type()` (return unique name). Building an EP requires adding files to ONNX Runtime source tree—**there is currently no plugin API** for out-of-tree development.

Academic projects using ONNX-based flows include FPGA accelerators through QONNX (quantized ONNX) and AMD's FINN/Vitis AI frameworks. The 2024 paper "ONNX-to-Hardware Design Flow for FPGA" demonstrates converting QONNX models to streaming FPGA accelerators with runtime-switchable inference configurations.

The integration path from PyTorch uses `torch.onnx.export(model, inputs, "model.onnx", dynamo=True)` for the modern export path, then `ort.InferenceSession("model.onnx", providers=['MyCustomEP', 'CPUExecutionProvider'])` for inference with fallback chain. ONNX Runtime's advantages include a mature ecosystem, framework-agnostic support (one EP serves PyTorch, TensorFlow, JAX models), and extensive graph optimizations.

Effort estimates: minimal skeleton EP in **2-4 weeks**, full-featured EP with comprehensive operator coverage in **6-12 months**. Reference implementations like CUDA EP (~15K LOC) and DNNL EP (~5K LOC) provide implementation patterns, though simpler EPs run ~1-2K LOC.

## MLIR-based flows maximize flexibility at higher implementation cost

MLIR provides multi-level intermediate representation with progressive lowering through custom dialects—ideal for research exploring novel compiler optimizations but requiring significant engineering investment. The key dialects for accelerators are **linalg** (perfect loop nests, tiling, fusion), **affine** (polyhedral compilation), and **vector** (SIMD operations).

**torch-mlir** serves as the entry point from PyTorch, offering three paths: TorchScript (most tested), Lazy Tensor Core (tracing-based), and FX/Dynamo (modern, pure Python). The backend contract normalizes torch dialect operations before lowering to linalg-on-tensors, TOSA, or StableHLO depending on target.

Projects like **SODA-OPT** demonstrate automatic kernel outlining, tiling, and pre-optimization for HLS backends—enabling direct FPGA/ASIC generation from PyTorch models. **AXI4MLIR** generates cache-optimal host drivers achieving 1.65× speedup over hand-optimized code. ONNX-MLIR provides plugin architecture for accelerators using `onnx_mlir::accel::Accelerator` base class.

NVDLA's software stack offers a complete reference: compiler (parses networks, generates optimized execution graphs, outputs "NVDLA Loadable" binary), User Mode Driver (processes loadables, binds tensors), and Kernel Mode Driver (engine scheduler, dependency execution). The loadable format standardizes across implementations, enabling separate compiler/runtime development—a pattern worth emulating.

For JAX users, **PJRT** (Pluggable Runtime) provides the standard interface for custom backends. Plugins register via namespace packages, package metadata, or environment variables. Intel reports integration taking **months** but benefiting from backward-compatible API working across JAX, TensorFlow, and PyTorch/XLA.

MLIR's advantages include framework-agnostic design (one backend serves torch-mlir, ONNX-MLIR, TensorFlow), progressive lowering preserving optimization opportunities, and strong industry adoption. Disadvantages include steep learning curve (TableGen, complex pass infrastructure), significant code overhead for custom dialects, and version churn requiring maintenance across LLVM releases.

## Comparing integration approaches for different project stages

| Approach | Min. Infrastructure | Time to Basic Integration | Best For | Primary Limitation |
|----------|-------------------|---------------------------|----------|-------------------|
| **ExecuTorch Delegate** | Python partitioner + C++ runtime | 1-2 months | PyTorch-native, edge deployment | Newer ecosystem, less documentation |
| **ONNX Runtime EP** | C++ EP in ORT source tree | 2-4 months | Multi-framework, mature tooling | In-tree build required |
| **TVM BYOC** | Python annotations + codegen + runtime | 2-4 months | Auto-tuning, FPGA targets | Steep learning curve |
| **MLIR Custom Dialect** | Dialect definition + lowering passes | 3-6 months | Research flexibility, compiler work | Highest implementation effort |
| **Gemmini-style ONNX** | ONNX Runtime fork + C kernel library | 2-3 months | Systolic arrays, RISC-V | Tied to RISC-V ecosystem |

For projects requiring the **least infrastructure to start**, ExecuTorch delegates or ONNX Runtime EPs with CPU fallback provide the fastest path. For projects prioritizing **scalability as they mature**, TVM's ecosystem offers better auto-tuning and optimization infrastructure. The Gemmini approach scales exceptionally well for **RISC-V-based systolic arrays** specifically.

The critical decision point is whether your project is PyTorch-only (favors ExecuTorch), multi-framework (favors ONNX Runtime), or research-focused with compiler optimization goals (favors MLIR or TVM).

## Building a minimum viable stack in 10 months

Given RTL simulation working, ISA defined, and basic assembly kernels existing, the recommended implementation order prioritizes end-to-end visibility over bottom-up foundation building:

**Months 1-2**: Lock down hardware/software interface specification (memory map, register interface, ISA). Set up Verilator-based simulation with waveform dumping for debugging. Validate 3-5 core assembly kernels (matmul is critical—it dominates neural network workload) against bit-accurate golden reference model in Python/C++.

**Months 3-4**: Build minimal code generator targeting your assembly kernels. Create basic memory allocation primitives and kernel launcher runtime. Focus on generating **correct code first**, not optimized code. Unit test each operator against PyTorch/NumPy reference outputs.

**Months 5-6**: Implement PyTorch integration via **PrivateUse1** mechanism (official pathway for out-of-tree backends since PyTorch 2.0) or ExecuTorch delegates. Register ~20-30 operators covering your target demo model. Implement CPU fallback for unsupported operations from day one—this enables partial execution even with incomplete operator coverage.

**Months 7-8**: Target model forward pass validation (MobileNet or ResNet-18 recommended—well-tested, representative workloads). Debug numerical mismatches methodically. Build automated regression suite running overnight on RTL simulation.

**Months 9-10**: Performance optimization on critical path only. Demo preparation. Stretch goals: batch processing, additional models.

The minimum operator set for CNN inference covers: matmul, conv2d, relu/relu6, maxpool, avgpool, batch normalization (can decompose to element-wise ops), add, flatten, linear. For transformer models, add: layer normalization, softmax, GELU, attention patterns.

## Key shortcuts and pitfalls from production projects

Successful accelerator projects share common pragmatic shortcuts. **Start with a known-working model**—torchvision's ResNet-18 or MobileNet-v2 have extensive testing and known numerical properties. **Use quantized models** (INT8) for simpler data types and easier validation. **Hardcode shapes initially**—dynamic shape support adds significant complexity better deferred until basic functionality works. **Skip autograd**—inference-only for first demo, add training support as stretch goal.

The most common pitfalls include underestimating memory subsystem complexity (data movement often dominates performance), premature optimization before functional correctness, and insufficient debug infrastructure. The Gemmini team emphasized that waveform access and register inspection were essential during bring-up—plan for debugging time at **2-3× your initial estimate** for numerical mismatch debugging specifically.

For RTL simulation development, use Verilator for single-operator kernel validation (fast compile, slow simulation at ~1-1000 kHz). Build a functional C++ model for compiler development iteration (fast execution). Run integration tests overnight on RTL simulation. Target FPGA emulation (FireSim, AWS F1) for full-model validation when RTL stabilizes—this provides **10-100 MHz** execution versus Verilator's kilohertz range.

## Conclusion

The research reveals that successful accelerator software stacks share a common architecture: a stable intermediate representation (ONNX, Relay, or MLIR dialect), a parameterized kernel library auto-tuned to hardware configuration, and multi-level programming interfaces serving different user expertise levels. The choice between ExecuTorch, TVM, ONNX Runtime, or MLIR depends primarily on project goals—native PyTorch integration, multi-framework support, auto-tuning capabilities, or research flexibility.

For a **10-month PyTorch demo on a systolic array accelerator**, the recommended path combines Gemmini's architectural lessons (parameterized C library, ONNX as IR) with ExecuTorch or PrivateUse1 for framework integration. Critical success factors include maintaining CPU fallback from day one, investing in automated testing infrastructure early, and treating debugging time as a first-class scheduling concern. The projects that succeed treat software as co-equal with hardware—starting compiler and framework integration in parallel with RTL development rather than sequentially after tape-out.