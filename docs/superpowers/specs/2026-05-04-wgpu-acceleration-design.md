# fem-rs wgpu Acceleration Design

> Version: 0.1.0 | 2026-05-04 | Status: Draft

## Motivation

Accelerate fem-rs computational kernels via wgpu (WebGPU) compute shaders.
Priorities (in order):

1. **Cross-platform**: macOS (Metal), Windows (DX12/Vulkan), Linux (Vulkan), browser (WebGPU/WASM)
2. **Pure Rust**: wgpu as the Rust-native GPU abstraction
3. **WASM/Web**: WebGPU is the only viable browser GPU API

## Approach: Hybrid

- **Matrix-free operators** → reed's wgpu backend (reed already has WGSL kernels for simplex basis evaluation, quadrature, Jacobian transforms)
- **Sparse linear algebra** → new `fem-linalg-gpu` crate with custom wgpu compute shaders (SpMV, vector ops)
- **Element assembly** → stays on CPU; CSR result uploaded to GPU once
- **Solvers** → GPU-resident iteration loop; only convergence scalar readback per iteration

---

## Crate Architecture

```
fem-rs/
├── crates/
│   ├── linalg/                    # existing: CPU CSR, COO, Vector
│   ├── linalg-gpu/                # ✨ NEW: GPU sparse linear algebra
│   │   ├── src/
│   │   │   ├── lib.rs             # re-exports, GpuContext
│   │   │   ├── buffer.rs          # DeviceBuffer<T>: owned+staged wgpu buffers
│   │   │   ├── csr.rs             # GpuCsrMatrix<T>: CSR on GPU
│   │   │   ├── vector.rs          # GpuVector<T>: dense vector on GPU
│   │   │   ├── spmv.rs            # spmv compute pipeline
│   │   │   ├── spmv.wgsl          # SpMV compute shader
│   │   │   ├── axpy.rs            # axpy, dot, norm2 pipelines
│   │   │   └── axpy.wgsl          # vector ops compute shader
│   │   └── Cargo.toml             # deps: wgpu, bytemuck, fem-core
│   │
│   ├── assembly/
│   │   └── src/reed/
│   │       ├── context.rs         # ✅ exists: FemCeed, CeedBackend::ReedGpuWgpu
│   │       └── context.wgpu.rs    # ✨ NEW: actual wgpu reed backend wiring
│   │
│   └── solver/
│       └── src/
│           ├── cg.rs              # ✅ exists: CPU CG
│           ├── cg_gpu.rs          # ✨ NEW: GPU-resident CG
│           ├── gmres_gpu.rs       # ✨ NEW: GPU-resident GMRES
│           └── precond/
│               └── jacobi_gpu.rs  # ✨ NEW: GPU Jacobi preconditioner
```

- `linalg-gpu` depends only on `fem-core` + `wgpu` (no circular deps)
- `fem-solver` gains optional GPU solver variants, conditionally depending on `fem-linalg-gpu`
- `fem-assembly/reed` already has `CeedBackend::ReedGpuWgpu` — made functional
- Feature gate: `gpu` in workspace, cascading through crates

---

## Data Flow & Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│  CPU                                                            │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Mesh    │───▶│  Assembler   │───▶│  CsrMatrix (CPU)     │  │
│  │  Space   │    │  (per-elem)  │    │  (COO → CSR)         │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                  │ upload once  │
│                                                  ▼              │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  Mesh    │───▶│  FemCeed     │───▶│  GpuCsrMatrix         │  │
│  │  reed    │    │  (mat-free)  │    │  GpuVector (rhs, sol) │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                  │              │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│──│
│  GPU                                               ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  GpuCgSolver / GpuGmresSolver                            │  │
│  │  ┌─────────┐   ┌──────────┐   ┌────────────────────┐    │  │
│  │  │ spmv    │◀──│ GpuCsr   │──▶│ GpuVector (x,y,r,p) │    │  │
│  │  │ (wgsl)  │   │ Matrix   │   │ all GPU-resident    │    │  │
│  │  └─────────┘   └──────────┘   └────────────────────┘    │  │
│  │       │              │                  │                 │  │
│  │       ▼              ▼                  ▼                 │  │
│  │  ┌─────────┐   ┌──────────┐   ┌────────────────────┐    │  │
│  │  │ axpy    │   │ Jacobi   │   │ dot / norm         │    │  │
│  │  │ (wgsl)  │   │ precond  │   │ (reduce pipeline)  │    │  │
│  │  └─────────┘   └──────────┘   └────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Convergence check: dot/norm → staging buffer → CPU read (1×)  │
└─────────────────────────────────────────────────────────────────┘
```

**Execution rules:**

1. **One-time upload**: CSR matrix + initial vectors uploaded at solver start
2. **Zero-copy iteration**: All SpMV, axpy, dot, preconditioner runs dispatch GPU compute passes, results stay in device buffers
3. **Single CPU sync per iteration**: Only the residual norm `||r||` is read back via staging buffer to check convergence
4. **Matrix-free path**: `FemCeed` applies `y = A x` directly on GPU via reed — no CSR storage needed, just geometry + quadrature data in uniform/storage buffers
5. **Fallback**: If no wgpu adapter available, transparently use existing CPU path

**Two solver families:**

| | Sparse (CSR) | Matrix-Free (reed) |
|---|---|---|
| **Setup** | Assemble CSR on CPU → upload | Upload mesh + build reed context on GPU |
| **Apply** | GpuCsrMatrix::spmv shader | reed operator apply (basis+grad shaders) |
| **Best for** | Low-order (P1/P2), repeated solves | High-order (P3+), memory-constrained |
| **Storage** | O(nnz) on GPU | O(n_qpts × n_fields) on GPU |

---

## GPU Type Design

### GpuContext

```rust
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    spmv_pipeline: SpmvPipeline,
    vector_pipeline: VectorOpsPipeline,
}

impl GpuContext {
    pub async fn new() -> Result<Self, GpuError>;
    pub fn device(&self) -> &wgpu::Device;
    pub fn queue(&self) -> &wgpu::Queue;
}
```

### GpuCsrMatrix

```rust
pub struct GpuCsrMatrix<T: Scalar> {
    nrows: u32,
    ncols: u32,
    nnz: u32,
    row_ptr: wgpu::Buffer,    // (nrows + 1) × u32
    col_idx: wgpu::Buffer,    // nnz × u32
    values: wgpu::Buffer,     // nnz × f32/f64
}

impl GpuCsrMatrix<f64> {
    pub fn from_cpu(ctx: &GpuContext, cpu: &CsrMatrix<f64>) -> Self;
    pub fn encode_spmv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x: &GpuVector<f64>,
        beta: f64,
        y: &mut GpuVector<f64>,
    );
}
```

### GpuVector

```rust
pub struct GpuVector<T: Scalar> {
    len: u32,
    buffer: wgpu::Buffer,
    staging: Option<wgpu::Buffer>,  // for CPU readback
    _marker: PhantomData<T>,
}

impl GpuVector<f64> {
    pub fn from_slice(ctx: &GpuContext, data: &[f64]) -> Self;
    pub fn zeros(ctx: &GpuContext, len: u32) -> Self;
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Vec<f64>;
    pub fn encode_axpy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x: &GpuVector<f64>,
        beta: f64,
        y: &mut GpuVector<f64>,
    );
    pub fn encode_dot(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        other: &GpuVector<f64>,
        ctx: &GpuContext,
    );
    pub fn encode_norm2(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &GpuContext,
    );
}
```

### Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| **f64 support** | Required | FEM needs double precision; check wgpu `f64` feature, fall back to emulated `f64` via two `u32` if unavailable |
| **Staging for dot/norm** | Use wgpu staging buffer + `map_async` | Avoids synchronous GPU→CPU stall; convergence check is async then read |
| **One CommandEncoder per iteration** | Encoder owns all passes for one solver step | Minimizes submission overhead; single `queue.submit()` per iteration |
| **Buffer lifecycle** | Caller owns buffers, solver borrows | Same GpuCsrMatrix reused across solves; GpuVectors reused across iterations |

### f64 strategy

wgpu's native `f64` is behind `wgpu::Features::SHADER_F64` — available on most desktop GPUs but not on WASM/WebGPU. Two paths:

1. **Desktop (Metal/Vulkan/DX12)**: use native `f64` in WGSL — zero overhead
2. **WASM fallback**: emulate f64 via `(u32, u32)` pairs with manual carry — slower but functional

Feature detection at `GpuContext::new()` time chooses the path.

---

## WGSL Shaders

### SpMV

1D workgroup dispatch over rows. Each workgroup processes a chunk of rows. Within a row, 8-way unrolled accumulation mirrors the CPU `csr_row_dot_f64` pattern.

```wgsl
// spmv.wgsl
struct Params {
    nrows: u32,
    alpha: f64,
    beta: f64,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read>  row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read>  col_idx: array<u32>;
@group(0) @binding(3) var<storage, read>  values: array<f64>;
@group(0) @binding(4) var<storage, read>  x: array<f64>;
@group(0) @binding(5) var<storage, read_write> y: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.nrows { return; }

    let start = row_ptr[row];
    let end = row_ptr[row + 1];

    // 8-way unrolled accumulation (same pattern as CPU csr_row_dot_f64)
    var sum: f64 = 0.0;
    let end8 = start + (end - start) / 8u * 8u;
    var k = start;
    while k < end8 {
        sum += values[k]     * x[col_idx[k]]
             + values[k + 1u] * x[col_idx[k + 1u]]
             + values[k + 2u] * x[col_idx[k + 2u]]
             + values[k + 3u] * x[col_idx[k + 3u]]
             + values[k + 4u] * x[col_idx[k + 4u]]
             + values[k + 5u] * x[col_idx[k + 5u]]
             + values[k + 6u] * x[col_idx[k + 6u]]
             + values[k + 7u] * x[col_idx[k + 7u]];
        k += 8u;
    }
    while k < end {
        sum += values[k] * x[col_idx[k]];
        k += 1u;
    }

    y[row] = params.alpha * sum + params.beta * y[row];
}
```

### Dot product / norm reduction

Two-pass reduction: first pass does per-workgroup partial sums via shared memory; second pass reduces workgroup results to a single scalar in a staging buffer.

### Jacobi preconditioner

Extract `A.diagonal()` → `GpuVector<f64>` once. Preconditioner step is element-wise divide: `z[i] = r[i] / diag[i]`. Same WGSL as axpy with additional divisor buffer.

---

## Implementation Phases

| Phase | Content | Effort |
|---|---|---|
| **P0: GPU linalg foundation** | `fem-linalg-gpu` crate; `GpuContext`, `GpuVector`, `GpuCsrMatrix`; upload/download; SpMV + axpy + dot WGSL shaders; f64 native path | 2-3 weeks |
| **P1: GPU solvers** | `GpuCgSolver`, `GpuGmresSolver` in `fem-solver`; Jacobi GPU precond; convergence check via staging readback; feature-gated behind `gpu` | 2-3 weeks |
| **P2: Reed wgpu backend** | Wire `FemCeed::ReedGpuWgpu` to actual reed wgpu backend; verify matrix-free operator application works; benchmark vs CPU | 2-3 weeks |
| **P3: Integration & optimization** | Unified `Solver` trait that auto-selects CPU/GPU; buffer pool for reuse; WASM f64 emulation path; GPU AMG (coarse levels on CPU) | 3-4 weeks |

### Feature gate chain

```toml
# workspace Cargo.toml
[features]
gpu = ["fem-linalg-gpu", "fem-solver/gpu", "fem-assembly/reed-gpu"]

# fem-assembly/Cargo.toml
[features]
reed-gpu = ["reed", "reed/wgpu", "fem-linalg-gpu"]
```

### Phase 0 acceptance criteria

- `GpuCsrMatrix::from_cpu` uploads a 1M DOF Laplacian in < 100ms
- `spmv` throughput ≥ 2× CPU SpMV on 1M DOF problem (desktop GPU)
- `GpuVector` dot product relative error < 1e-14 vs CPU
- `spmv` correctness: `||gpu_spmv(A,x) - cpu_spmv(A,x)||_∞ < 1e-14`

### Phase 1 acceptance criteria

- `GpuCgSolver` converges in same iteration count as CPU CG
- End-to-end solve wall time ≤ 0.5× CPU solve time for 1M DOF Poisson
- Feature gate: `cargo build -p fem-solver` (no gpu) does not link wgpu

### Phase 2 acceptance criteria

- `FemCeed::apply_poisson_2d` with `ReedGpuWgpu` backend produces results matching CPU to 1e-13
- Matrix-free operator throughput ≥ CPU CSR SpMV for Q4+ element order

### Phase 3 acceptance criteria

- WASM target: `cargo build --target wasm32-unknown-unknown -p fem-wasm --features gpu` produces working WebGPU path
- GpuAmgSolver V-cycle convergence factor < 0.3 for Laplacian
- Uncached first solve latency (incl. GPU upload) ≤ 3× CPU solve for 1M DOF

---

## f64 Support Details

### Detection

```rust
impl GpuContext {
    pub async fn new() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: if has_f64 {
                        wgpu::Features::SHADER_F64
                    } else {
                        wgpu::Features::empty()
                    },
                    ..Default::default()
                },
                None,
            )
            .await?;

        // Select shader variant based on has_f64
        // ...
    }
}
```

### WASM emulated f64

When `SHADER_F64` is unavailable (WASM), represent f64 as `vec2<u32>`:
- `pack_f64(v: f64) -> vec2<u32>`: bitcast via `bitcast<u64>(v)`, split hi/lo
- `unpack_f64(p: vec2<u32>) -> f64`: reconstruct u64, bitcast back
- Arithmetic: manual carry propagation for add/sub; split-mantissa for mul

This path is 4-8× slower than native f64 but functional.

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| **f64 on WASM too slow** | Acceptable: WASM GPU path is for demo/small problems; desktop uses native f64 |
| **reed wgpu backend incomplete/missing** | Phase 2 starts with a reed wgpu feasibility check. If reed lacks functional wgpu support or the feature gate name differs from assumed `reed/wgpu`, GPU work focuses on sparse linalg + solvers (P0-P1); matrix-free GPU path deferred to later reed releases |
| **reed dependency stale/broken** | Current reed is pinned at a specific git rev. Before P2, update to latest reed main and verify wgpu backend status |
| **GPU memory limits** | For >1B DOF, GPU sparse storage won't fit; matrix-free path handles this case |
| **Pipeline compilation latency** | Pre-compile shaders at `GpuContext::new()` time; cache if possible |
| **Apple GPU tile-memory constraints** | Workgroup size 256 is safe for Apple GPUs; tune per-platform if needed |
