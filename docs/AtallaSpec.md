# Atalla Spec (Phase 1, Draft)

This central spec replaces the standalone ABI, blob format, scratchpad, ISA
mapping, and op‑set docs.

## Phase 1 Locks
- Row‑major contiguous tensors only.
- Softmax only on last dimension.
- Output tensors are runtime‑allocated.

## Kernel ABI (C)
### Conventions
- All tensors are BF16 unless noted.
- Strides are in elements (not bytes) and must match contiguous layout.
- Alignment: 64 bytes for DRAM pointers; 256 bytes for scratchpad pointers.
- Error handling: kernels return 0 on success, nonzero on failure.

### Core Types
```c
typedef uint16_t bf16;

typedef struct {
    bf16* data;
    int32_t ndim;
    int32_t sizes[4];
    int32_t strides[4];
} atalla_tensor;
```

### Kernel Signatures
```c
int atalla_gemm(const atalla_tensor* a,
                const atalla_tensor* b,
                atalla_tensor* out);

int atalla_relu(const atalla_tensor* x,
                atalla_tensor* out);

int atalla_softmax(const atalla_tensor* x,
                   int32_t dim,
                   atalla_tensor* out);

int atalla_add(const atalla_tensor* a,
               const atalla_tensor* b,
               atalla_tensor* out);

int atalla_sub(const atalla_tensor* a,
               const atalla_tensor* b,
               atalla_tensor* out);

int atalla_mul(const atalla_tensor* a,
               const atalla_tensor* b,
               atalla_tensor* out);

int atalla_transpose(const atalla_tensor* x,
                     int32_t dim0,
                     int32_t dim1,
                     atalla_tensor* out);
```

### Notes
- Broadcast is not supported in Phase 1; shapes must match exactly.
- Layout transforms are handled in the compiler, not in kernels.

## Scratchpad Allocation API
### C Interface
```c
void* scpad_alloc(size_t bytes, size_t align, int* sid);
void  scpad_free(void* ptr, int sid);
```

### Semantics
- `scpad_alloc` returns a scratchpad pointer aligned to `align` and sets `sid`.
- `align` must be a power of two; minimum 64 bytes.
- Allocation unit is a page (implementation‑defined).
- If allocation fails, returns `NULL`.
- `scpad_free(NULL, sid)` is a no‑op.

### Constraints
- Scratchpad is banked (2 banks).
- Tiles are 32x32 BF16 (2048 bytes per tile).
- SDMA loads/stores operate on full tiles only.
- VM loads/stores operate on vector slices extracted from tiles.

### SDMA Load/Store Semantics
- `scpad.ld` loads a tile from GMEM to scratchpad.
- `scpad.st` stores a tile from scratchpad to GMEM.
- Fields:
  - `sid`: scratchpad bank id (0 or 1).
  - `rs2`: register holding GMEM base address.
  - `rs1`/`rd1`: register holding scratchpad base address (dependency).

### Instruction Encoding (Reference)
- `scpad.ld` opcode `1011000`
- `scpad.st` opcode `1011001`
- SDMA fields: `sid`, `num_rows`, `num_cols`, `rs2`, `rs1/rd1`, `opcode[6:0]`

### VM Vector Load/Store (Reference)
- `vreg.ld` opcode `1001101`
- `vreg.st` opcode `1001110`
- VM fields: `rc_id`, `rc`, `sid`, `num_rows`, `num_cols`, `rs1`, `vd`,
  `opcode[6:0]`

## ISA Mapping Summary
### GEMM (C = A × B)
**Goal:** stream vectors into the systolic array using `gemm.vv`.

**Sequence (per K tile):**
1. `scpad.ld` A tile (GMEM → SCPAD sid0)
2. `scpad.ld` B tile (GMEM → SCPAD sid1)
3. `vreg.ld` B vectors into vector regs (weight preload)
4. For each A row vector:
   - `vreg.ld` A row vector
   - `gemm.vv` stream A + weights into systolic array
   - `vreg.st` store output vector into C tile buffer
5. `scpad.st` C tile (SCPAD → GMEM)

### Transpose (Xᵀ)
**Sequence (per tile):**
1. `scpad.ld` load tile into scratchpad
2. For each `rc_id` (0..31):
   - `vreg.ld` with `rc = row` into `vd`
   - `vreg.st` with `rc = col` into destination tile
3. `scpad.st` store transposed tile back to GMEM

### Softmax (row‑wise)
**Sequence (per row vector):**
1. `vreg.ld` load row vector
2. `rmax.vi` reduce max
3. `sub.vs` subtract max (broadcast scalar)
4. `expi.vi` elementwise exp
5. `rsum.vi` reduce sum
6. `div.vs` divide by sum
7. `vreg.st` store row vector

### Elementwise (add/sub/mul)
1. `vreg.ld` A, `vreg.ld` B
2. `add.vv` / `sub.vv` / `mul.vv`
3. `vreg.st` output

### ReLU (draft)
1. `vreg.ld` X
2. `mgt.vs` compare X > 0 to mask
3. `add.vv` or `mul.vv` with mask to zero‑out negatives
4. `vreg.st`

## Program Blob Format (Draft)
### Header
```
struct atalla_blob_header {
    char     magic[8];      // "ATALLA00"
    uint32_t version;       // 1
    uint32_t op_count;      // number of ops
    uint32_t const_bytes;   // size of constants section
    uint32_t meta_bytes;    // size of metadata section
};
```

### Sections
1. **Ops**: linear list of ops to execute.
2. **Constants**: packed weights (bf16) and scalars.
3. **Metadata**: tensor shapes, strides, and layout tags.

### Op Encoding (Phase 1)
```
enum atalla_op_kind {
    OP_GEMM = 1,
    OP_RELU = 2,
    OP_SOFTMAX = 3,
    OP_ADD = 4,
    OP_SUB = 5,
    OP_MUL = 6,
    OP_TRANSPOSE = 7,
};

struct atalla_op {
    uint32_t kind;
    uint32_t input_ids[2];
    uint32_t output_id;
    int32_t  dim0;          // transpose dim0; softmax dim in dim0
    int32_t  dim1;          // transpose dim1; otherwise -1
};
```

### Metadata (Phase 1)
Each tensor has:
- `ndim`
- `sizes[]`
- `strides[]`
- `dtype` (bf16)
- `storage` (dram or scratchpad)

### Versioning
Any incompatible change increments `version`. Backwards compatibility is not
guaranteed in v1.

## Model Op Sets (Phase 1)
### AttentionBlock (no conv)
- `aten.matmul.default`
- `aten.transpose.int`
- `aten.softmax.int`

### TinyTransformerBlock (no conv)
- `aten.matmul.default`
- `aten.transpose.int`
- `aten.softmax.int`
- `aten.relu.default`
- `aten.add.Tensor`

### ElementwiseMix
- `aten.add.Tensor`
- `aten.sub.Tensor`
- `aten.mul.Tensor`

## Pending Confirmations
### Blob Format / Metadata
- [ ] Final blob header fields (magic, version, sizes).
- [ ] Tensor metadata: shapes, strides, dtype tags.
- [ ] Constant/weight section layout and alignment.
- [ ] Op encoding schema (inputs/outputs, dim fields).

### ISA Semantics
- [ ] `gemm.vv` weight preload protocol (`lw.vi` or equivalent).
- [ ] ReLU mask lane behavior (masked write semantics).
- [ ] SDMA swizzle function and layout constraints.

### Runtime Rules
- [ ] Scratchpad allocator policy (page size, reuse, failure).
- [ ] Softmax axis policy (last‑dim only confirmed).
- [ ] Transpose constraints (allowed dims, tile size limits).

### Compiler/Kernel Interface
- [ ] Output buffer ownership (runtime alloc, kernel writes).
- [ ] Error codes and failure semantics.

## References
- `AtallaISA.csv`
- `ISA Atalla Bit-Spec.csv`
