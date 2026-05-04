# wgpu Acceleration Implementation Plan (Phase 0 + Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** GPU-accelerated sparse linear algebra (SpMV, vector ops) and iterative solvers (CG, GMRES) via wgpu compute shaders.

**Architecture:** New `fem-linalg-gpu` crate holds GPU types (GpuContext, GpuCsrMatrix, GpuVector) and WGSL compute shaders for SpMV/axpy/dot/norm2. New `fem-solver` GPU path implements CG and GMRES iteration loops entirely on GPU, reading back only the convergence scalar per iteration. Existing CPU path is unchanged behind a `gpu` feature gate.

**Tech Stack:** wgpu 24.x (cross-platform WebGPU), bytemuck (Pod casting), WGSL compute shaders, tokio or pollster (async init), fem-core (Scalar trait), fem-linalg (CPU CSR for upload source)

---

## File Structure

```
crates/linalg-gpu/                    ✨ NEW crate
├── Cargo.toml
└── src/
    ├── lib.rs                        re-exports, GpuError, feature detection
    ├── context.rs                    GpuContext: device, queue, pipelines
    ├── buffer.rs                     DeviceBuffer: wgpu::Buffer + staging helpers
    ├── csr.rs                        GpuCsrMatrix<T>: GPU-resident CSR
    ├── vector.rs                     GpuVector<T>: GPU-resident dense vector
    ├── spmv_pipeline.rs              SpMV compute pipeline + WGSL module
    ├── vector_pipeline.rs            axpy / dot / norm2 compute pipelines
    ├── spmv.wgsl                     SpMV compute shader (f64 native)
    └── vector_ops.wgsl               axpy, dot-reduce, norm2-reduce shaders

crates/solver/
├── Cargo.toml                        ✨ MODIFY: add gpu feature + dep on linalg-gpu
└── src/
    ├── lib.rs                        ✨ MODIFY: add gpu module re-export
    ├── cg_gpu.rs                     ✨ NEW: GPU-resident CG solver
    └── gmres_gpu.rs                  ✨ NEW: GPU-resident GMRES solver
```

**Dependency graph:**
```
fem-linalg-gpu → fem-core, wgpu, bytemuck
fem-solver (gpu) → fem-linalg-gpu, fem-linalg, fem-core
```

---

### Task 1: Create fem-linalg-gpu crate skeleton

**Files:**
- Create: `crates/linalg-gpu/Cargo.toml`
- Create: `crates/linalg-gpu/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Create Cargo.toml for fem-linalg-gpu**

```toml
# crates/linalg-gpu/Cargo.toml
[package]
name    = "fem-linalg-gpu"
version = "0.1.0"
edition = "2021"

[dependencies]
fem-core   = { path = "../core" }
bytemuck   = { workspace = true }
wgpu       = "24"
pollster   = "0.4"
thiserror  = "2"
log        = { workspace = true }

[lib]
crate-type = ["lib"]
```

- [ ] **Step 2: Create lib.rs with module declarations and GpuError**

```rust
// crates/linalg-gpu/src/lib.rs
pub mod buffer;
pub mod context;
pub mod csr;
pub mod spmv_pipeline;
pub mod vector;
pub mod vector_pipeline;

pub use buffer::DeviceBuffer;
pub use context::GpuContext;
pub use csr::GpuCsrMatrix;
pub use vector::GpuVector;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable wgpu adapter found")]
    NoAdapter,
    #[error("wgpu device request failed: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("buffer creation failed: {0}")]
    Buffer(#[from] wgpu::BufferAsyncError),
    #[error("f64 not supported by GPU and no emulation path compiled")]
    F64Unavailable,
}

pub type GpuResult<T> = Result<T, GpuError>;
```

- [ ] **Step 3: Register the crate in workspace Cargo.toml**

Edit `Cargo.toml` line after `"crates/linalg"`:
```toml
"crates/linalg-gpu",
```

- [ ] **Step 4: Verify crate skeleton compiles**

Run: `cargo check -p fem-linalg-gpu 2>&1`
Expected: `Checking fem-linalg-gpu...` success (unused import warnings for stub modules OK)

- [ ] **Step 5: Commit**

```bash
git add crates/linalg-gpu/ Cargo.toml
git commit -m "feat: fem-linalg-gpu crate skeleton"
```

---

### Task 2: DeviceBuffer — wgpu buffer with staging

**Files:**
- Create: `crates/linalg-gpu/src/buffer.rs`

- [ ] **Step 1: Write the failing doc-test in buffer.rs**

We'll test by running `cargo test` on a doc-test that exercises buffer creation. Write the module skeleton first, then add a test.

```rust
// crates/linalg-gpu/src/buffer.rs
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Owned wgpu buffer with optional staging for readback.
///
/// Wraps a `wgpu::Buffer` and provides helper methods to upload slices
/// of `bytemuck::Pod` data and read back via a staging buffer.
pub struct DeviceBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    /// Present only when readback was requested.
    staging: Option<wgpu::Buffer>,
}

impl DeviceBuffer {
    /// Create a buffer with the given size and usage, initialized to zero.
    pub fn new(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self { buffer, size, usage, staging: None }
    }

    /// Create a buffer from a byte slice (one-shot upload via staging).
    pub fn from_bytes<T: bytemuck::Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[T],
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        });
        Self {
            buffer,
            size: (data.len() * std::mem::size_of::<T>()) as u64,
            usage,
            staging: None,
        }
    }

    /// Create with a staging buffer for CPU readback.
    pub fn with_staging(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{label}_buffer").as_str()),
            size,
            usage,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(format!("{label}_staging").as_str()),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            usage,
            staging: Some(staging),
        }
    }

    /// Raw wgpu buffer reference.
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buffer }

    /// Size in bytes.
    pub fn size(&self) -> u64 { self.size }

    /// Staging buffer reference (panics if not created with `with_staging`).
    pub fn staging(&self) -> &wgpu::Buffer {
        self.staging.as_ref().expect("DeviceBuffer has no staging buffer")
    }

    /// Encode a copy from self to the staging buffer.
    pub fn encode_copy_to_staging(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(ref staging) = self.staging {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, staging, 0, self.size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (wgpu::Device, wgpu::Queue) {
        pollster::block_on(async {
            let instance = wgpu::Instance::default();
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .expect("need a wgpu adapter");
            adapter
                .request_device(&wgpu::DeviceDescriptor::default(), None)
                .await
                .expect("need a wgpu device")
        })
    }

    #[test]
    fn buffer_from_f64_slice() {
        let (device, queue) = test_device();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = DeviceBuffer::from_bytes(
            &device, &queue, &data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "test_f64",
        );
        assert_eq!(buf.size(), 4 * 8);
    }

    #[test]
    fn buffer_with_staging_has_staging() {
        let (device, _queue) = test_device();
        let buf = DeviceBuffer::with_staging(
            &device,
            1024,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "test_staging",
        );
        let _staging = buf.staging(); // does not panic
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p fem-linalg-gpu -- --test-threads=1 2>&1`
Expected: 2 tests pass (buffer_from_f64_slice, buffer_with_staging_has_staging)

- [ ] **Step 3: Commit**

```bash
git add crates/linalg-gpu/src/buffer.rs
git commit -m "feat: DeviceBuffer with staging support"
```

---

### Task 3: GpuContext — device, queue, and pipeline initialization

**Files:**
- Create: `crates/linalg-gpu/src/context.rs`
- Modify: `crates/linalg-gpu/src/lib.rs` (re-export GpuContext)

- [ ] **Step 1: Write GpuContext**

```rust
// crates/linalg-gpu/src/context.rs
use std::sync::Arc;
use wgpu::Device;

/// Manages the wgpu device, queue, and pre-compiled compute pipelines.
///
/// Created once per process via [`GpuContext::new()`] (async). Pipelines are
/// compiled at construction time to avoid latency during solve iterations.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub features: GpuFeatures,
}

/// Detected GPU capabilities that affect shader selection.
#[derive(Debug, Clone)]
pub struct GpuFeatures {
    /// Native `f64` in WGSL (`wgpu::Features::SHADER_F64`).
    pub native_f64: bool,
    /// Maximum storage buffer size.
    pub max_buffer_size: u64,
    /// Maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dim: u32,
}

impl GpuContext {
    /// Initialize wgpu, select adapter, create device, and detect features.
    pub async fn new() -> Result<Self, crate::GpuError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(crate::GpuError::NoAdapter)?;

        let features = adapter.features();
        let has_f64 = features.contains(wgpu::Features::SHADER_F64);
        let limits = adapter.limits();

        let required_features = if has_f64 {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("fem-linalg-gpu"),
                    required_features,
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .await?;

        let gpu_features = GpuFeatures {
            native_f64: has_f64,
            max_buffer_size: limits.max_storage_buffer_binding_size as u64,
            max_compute_workgroups_per_dim: limits.max_compute_workgroups_per_dim,
        };

        if !has_f64 {
            log::warn!("SHADER_F64 not available; f64 emulation path not yet implemented — f64 buffers will error");
        }

        Ok(Self { device, queue, features: gpu_features })
    }

    /// Synchronous init using pollster (for non-async contexts).
    pub fn new_sync() -> Result<Self, crate::GpuError> {
        pollster::block_on(Self::new())
    }
}
```

- [ ] **Step 2: Update lib.rs to re-export context types**

Add after existing `pub use context::GpuContext;` line (already present from skeleton):
```rust
// (already in lib.rs from Task 1)
pub use context::{GpuContext, GpuFeatures};
```

Edit `crates/linalg-gpu/src/lib.rs` to ensure the re-export includes GpuFeatures:

```rust
pub use context::{GpuContext, GpuFeatures};
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p fem-linalg-gpu 2>&1`
Expected: success (warnings OK)

- [ ] **Step 4: Commit**

```bash
git add crates/linalg-gpu/src/context.rs crates/linalg-gpu/src/lib.rs
git commit -m "feat: GpuContext with adapter selection and f64 detection"
```

---

### Task 4: GpuVector — GPU-resident dense vector

**Files:**
- Create: `crates/linalg-gpu/src/vector.rs`

- [ ] **Step 1: Write GpuVector with upload, readback, and encode helpers**

```rust
// crates/linalg-gpu/src/vector.rs
use std::marker::PhantomData;
use fem_core::Scalar;
use crate::buffer::DeviceBuffer;
use crate::GpuContext;

/// Dense vector resident on the GPU.
pub struct GpuVector<T: Scalar> {
    len: u32,
    buffer: DeviceBuffer,
    _marker: PhantomData<T>,
}

impl<T: Scalar> GpuVector<T> {
    /// Create a zero-initialized vector of length `len` on the GPU.
    pub fn zeros(ctx: &GpuContext, len: u32) -> Self {
        let size = len as u64 * std::mem::size_of::<T>() as u64;
        let buffer = DeviceBuffer::new(
            &ctx.device,
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "gpu_vector",
        );
        // Zero fill via queue write
        let zeros = vec![0u8; size as usize];
        ctx.queue.write_buffer(buffer.buffer(), 0, &zeros);
        Self { len, buffer, _marker: PhantomData }
    }

    /// Upload from a CPU slice.
    pub fn from_slice(ctx: &GpuContext, data: &[T]) -> Self {
        let buffer = DeviceBuffer::from_bytes(
            &ctx.device,
            &ctx.queue,
            data,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "gpu_vector",
        );
        Self {
            len: data.len() as u32,
            buffer,
            _marker: PhantomData,
        }
    }

    /// Read back to CPU. Blocks until the copy completes.
    /// Only use for convergence checks (once per iteration).
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Vec<T> {
        let size = self.len as u64 * std::mem::size_of::<T>() as u64;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_staging"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(self.buffer.buffer(), 0, &staging, 0, size);
        ctx.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    /// Length of the vector.
    pub fn len(&self) -> u32 { self.len }

    /// Raw buffer reference for pipeline binding.
    pub fn buffer(&self) -> &wgpu::Buffer { self.buffer.buffer() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuContext;

    fn ctx() -> GpuContext {
        GpuContext::new_sync().expect("gpu context")
    }

    #[test]
    fn zeros_is_zero() {
        let gpu = ctx();
        let v: GpuVector<f64> = GpuVector::zeros(&gpu, 10);
        let cpu = v.read_to_cpu(&gpu);
        assert_eq!(cpu.len(), 10);
        for &x in &cpu { assert_eq!(x, 0.0); }
    }

    #[test]
    fn from_slice_roundtrip() {
        let gpu = ctx();
        let data: Vec<f64> = vec![1.0, 2.0, 3.14159, -5.0];
        let v = GpuVector::from_slice(&gpu, &data);
        let cpu = v.read_to_cpu(&gpu);
        assert_eq!(cpu.len(), data.len());
        for (a, b) in data.iter().zip(cpu.iter()) {
            assert!((a - b).abs() < 1e-15, "mismatch: {a} vs {b}");
        }
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p fem-linalg-gpu -- --test-threads=1 2>&1`
Expected: 4 tests pass (2 from buffer + 2 from vector)

- [ ] **Step 3: Commit**

```bash
git add crates/linalg-gpu/src/vector.rs
git commit -m "feat: GpuVector with upload and readback"
```

---

### Task 5: GpuCsrMatrix — GPU-resident CSR sparse matrix

**Files:**
- Create: `crates/linalg-gpu/src/csr.rs`

- [ ] **Step 1: Write GpuCsrMatrix**

```rust
// crates/linalg-gpu/src/csr.rs
use std::marker::PhantomData;
use fem_core::Scalar;
use crate::buffer::DeviceBuffer;
use crate::GpuContext;

/// CSR sparse matrix resident on the GPU.
///
/// Stores `row_ptr` (u32), `col_idx` (u32), and `values` (T) in separate
/// wgpu storage buffers.
pub struct GpuCsrMatrix<T: Scalar> {
    pub nrows: u32,
    pub ncols: u32,
    pub nnz: u32,
    row_ptr: DeviceBuffer,
    col_idx: DeviceBuffer,
    values: DeviceBuffer,
    _marker: PhantomData<T>,
}

impl<T: Scalar> GpuCsrMatrix<T> {
    /// Upload a CPU CSR matrix to the GPU.
    ///
    /// Converts `row_ptr` from `usize` to `u32` (panics if any value exceeds
    /// `u32::MAX` — safe for DOF counts under 4 billion).
    pub fn from_cpu(
        ctx: &GpuContext,
        cpu: &fem_linalg::CsrMatrix<T>,
    ) -> Self {
        let row_ptr_u32: Vec<u32> = cpu.row_ptr.iter().map(|&x| {
            assert!(x <= u32::MAX as usize, "row_ptr value exceeds u32 max");
            x as u32
        }).collect();
        let col_idx_u32: Vec<u32> = cpu.col_idx.clone();

        let row_ptr_buf = DeviceBuffer::from_bytes(
            &ctx.device, &ctx.queue,
            &row_ptr_u32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "csr_row_ptr",
        );
        let col_idx_buf = DeviceBuffer::from_bytes(
            &ctx.device, &ctx.queue,
            &col_idx_u32,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "csr_col_idx",
        );
        let values_buf = DeviceBuffer::from_bytes(
            &ctx.device, &ctx.queue,
            cpu.values.as_slice(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "csr_values",
        );

        Self {
            nrows: cpu.nrows as u32,
            ncols: cpu.ncols as u32,
            nnz: cpu.values.len() as u32,
            row_ptr: row_ptr_buf,
            col_idx: col_idx_buf,
            values: values_buf,
            _marker: PhantomData,
        }
    }

    pub fn row_ptr_buffer(&self) -> &wgpu::Buffer { self.row_ptr.buffer() }
    pub fn col_idx_buffer(&self) -> &wgpu::Buffer { self.col_idx.buffer() }
    pub fn values_buffer(&self)  -> &wgpu::Buffer { self.values.buffer() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fem_linalg::CsrMatrix;

    fn ctx() -> GpuContext {
        GpuContext::new_sync().expect("gpu context")
    }

    /// Build a tiny 3×3 CSR matrix:
    /// [2 0 1]
    /// [0 3 0]
    /// [1 0 4]
    fn tiny_csr() -> CsrMatrix<f64> {
        let nrows = 3;
        let ncols = 3;
        let row_ptr = vec![0, 2, 3, 5];
        let col_idx = vec![0u32, 2, 1, 0, 2];
        let values = vec![2.0, 1.0, 3.0, 1.0, 4.0];
        CsrMatrix { nrows, ncols, row_ptr, col_idx, values }
    }

    #[test]
    fn from_cpu_preserves_dims() {
        let gpu = ctx();
        let cpu = tiny_csr();
        let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu);
        assert_eq!(gpu_mat.nrows, 3);
        assert_eq!(gpu_mat.ncols, 3);
        assert_eq!(gpu_mat.nnz, 5);
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p fem-linalg-gpu -- --test-threads=1 2>&1`
Expected: 5 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/linalg-gpu/src/csr.rs
git commit -m "feat: GpuCsrMatrix for GPU-resident CSR storage"
```

---

### Task 6: SpMV compute pipeline (Rust side)

**Files:**
- Create: `crates/linalg-gpu/src/spmv.wgsl`
- Create: `crates/linalg-gpu/src/spmv_pipeline.rs`
- Modify: `crates/linalg-gpu/src/lib.rs` (add module)

- [ ] **Step 1: Write the SpMV WGSL shader**

```wgsl
// crates/linalg-gpu/src/spmv.wgsl
struct Params {
    alpha: f64,
    beta: f64,
    nrows: u32,
    _pad: u32,  // align to 16 bytes after 3 u32
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
    let end = row_ptr[row + 1u];

    var sum: f64 = 0.0;
    let n = end - start;
    let end8 = start + (n / 8u) * 8u;
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

- [ ] **Step 2: Write the SpMV pipeline module**

```rust
// crates/linalg-gpu/src/spmv_pipeline.rs
use std::borrow::Cow;
use fem_core::Scalar;
use crate::{GpuContext, GpuCsrMatrix, GpuVector, GpuError};

/// SpMV parameter uniform layout: alpha (f64), beta (f64), nrows (u32), _pad (u32)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpmvParams {
    alpha: f64,
    beta: f64,
    nrows: u32,
    _pad: u32,
}

/// Pre-compiled SpMV compute pipeline.
pub struct SpmvPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SpmvPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_source = if native_f64 {
            Cow::Borrowed(include_str!("spmv.wgsl"))
        } else {
            panic!("f64 emulation path not yet implemented");
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("spmv_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("spmv_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("spmv_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("spmv_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
        });

        Self { pipeline, bind_group_layout }
    }

    /// Encode an SpMV dispatch: `y = alpha * A * x + beta * y`.
    pub fn encode_spmv<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        mat: &GpuCsrMatrix<T>,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        // Uniform params buffer
        let params = SpmvParams {
            alpha,
            beta,
            nrows: mat.nrows,
            _pad: 0,
        };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spmv_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spmv_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: mat.row_ptr_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: mat.col_idx_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: mat.values_buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroup_count = (mat.nrows + 255) / 256;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("spmv_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
    }

    /// Compute `y = A * x` (convenience shorthand).
    pub fn spmv<T: Scalar>(
        &self,
        ctx: &GpuContext,
        alpha: f64,
        mat: &GpuCsrMatrix<T>,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_spmv(ctx, &mut encoder, alpha, mat, x, beta, y);
        ctx.queue.submit(Some(encoder.finish()));
    }
}
```

- [ ] **Step 3: Update lib.rs**

Edit `crates/linalg-gpu/src/lib.rs`:
```rust
pub use spmv_pipeline::SpmvPipeline;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p fem-linalg-gpu 2>&1`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add crates/linalg-gpu/src/spmv.wgsl crates/linalg-gpu/src/spmv_pipeline.rs crates/linalg-gpu/src/lib.rs
git commit -m "feat: SpMV compute pipeline with WGSL shader"
```

---

### Task 7: SpMV correctness test

**Files:**
- Create: `crates/linalg-gpu/tests/spmv_test.rs`

- [ ] **Step 1: Write the integration test**

```rust
// crates/linalg-gpu/tests/spmv_test.rs
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{GpuContext, GpuCsrMatrix, GpuVector, SpmvPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

/// 3×3 SPD matrix, manually verified.
fn tiny_spd() -> CsrMatrix<f64> {
    CsrMatrix {
        nrows: 3,
        ncols: 3,
        row_ptr: vec![0, 2, 3, 5],
        col_idx: vec![0u32, 2, 1, 0, 2],
        values: vec![2.0, 1.0, 3.0, 1.0, 4.0],
    }
}

fn cpu_spmv(a: &CsrMatrix<f64>, x: &[f64], y: &mut [f64]) {
    for row in 0..a.nrows {
        let start = a.row_ptr[row];
        let end = a.row_ptr[row + 1];
        let mut s = 0.0;
        for k in start..end {
            s += a.values[k] * x[a.col_idx[k] as usize];
        }
        y[row] = s;
    }
}

#[test]
fn spmv_matches_cpu() {
    let gpu = ctx();
    let cpu_mat = tiny_spd();
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu_mat);
    let x = GpuVector::from_slice(&gpu, &[1.0, 2.0, 3.0]);
    let gpu_y = GpuVector::<f64>::zeros(&gpu, 3);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    pipeline.spmv(&gpu,
        1.0, &gpu_mat, &x,
        0.0, &gpu_y,
    );

    let gpu_result = gpu_y.read_to_cpu(&gpu);

    let mut cpu_result = vec![0.0; 3];
    cpu_spmv(&cpu_mat, &[1.0, 2.0, 3.0], &mut cpu_result);

    for i in 0..3 {
        let diff = (gpu_result[i] - cpu_result[i]).abs();
        assert!(diff < 1e-14, "row {i}: gpu={} cpu={} diff={}", gpu_result[i], cpu_result[i], diff);
    }
}

#[test]
fn spmv_with_alpha_beta() {
    let gpu = ctx();
    let cpu_mat = tiny_spd();
    let gpu_mat = GpuCsrMatrix::<f64>::from_cpu(&gpu, &cpu_mat);
    let x = GpuVector::from_slice(&gpu, &[1.0, 0.0, 0.0]);
    // Start y = [2, 2, 2]
    let gpu_y = GpuVector::from_slice(&gpu, &[2.0, 2.0, 2.0]);

    let pipeline = SpmvPipeline::new(&gpu.device, gpu.features.native_f64);
    // y = 3*A*x + 0.5*y
    pipeline.spmv(&gpu,
        3.0, &gpu_mat, &x,
        0.5, &gpu_y,
    );

    let gpu_result = gpu_y.read_to_cpu(&gpu);

    // A*x (column 0) = [2, 0, 1]
    // y = 3*[2,0,1] + 0.5*[2,2,2] = [6+1, 0+1, 3+1] = [7, 1, 4]
    assert!((gpu_result[0] - 7.0).abs() < 1e-14);
    assert!((gpu_result[1] - 1.0).abs() < 1e-14);
    assert!((gpu_result[2] - 4.0).abs() < 1e-14);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p fem-linalg-gpu --test spmv_test -- --test-threads=1 2>&1`
Expected: 2 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/linalg-gpu/tests/
git commit -m "test: SpMV correctness against CPU reference"
```

---

### Task 8: Vector operations pipeline (axpy, dot, norm2)

**Files:**
- Create: `crates/linalg-gpu/src/vector_ops.wgsl`
- Create: `crates/linalg-gpu/src/vector_pipeline.rs`

- [ ] **Step 1: Write vector_ops WGSL shader**

```wgsl
// crates/linalg-gpu/src/vector_ops.wgsl

// ── Axpy: y = alpha * x + beta * y ──────────────────────────────
struct AxpyParams {
    alpha: f64,
    beta: f64,
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read>  axpy_x: array<f64>;
@group(0) @binding(2) var<storage, read_write> axpy_y: array<f64>;

@compute @workgroup_size(256)
fn axpy_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= axpy_params.len { return; }
    axpy_y[i] = axpy_params.alpha * axpy_x[i] + axpy_params.beta * axpy_y[i];
}

// ── Dot product: workgroup-local reduction ──────────────────────
struct DotParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<uniform> dot_params: DotParams;
@group(1) @binding(1) var<storage, read> dot_a: array<f64>;
@group(1) @binding(2) var<storage, read> dot_b: array<f64>;
@group(1) @binding(3) var<storage, read_write> dot_result: array<f64>;

var<workgroup> wg_dot: array<f64, 256>;

@compute @workgroup_size(256)
fn dot_main(@builtin(local_invocation_id) lid: u32,
            @builtin(global_invocation_id) gid: u32,
            @builtin(num_workgroups) num_groups: u32) {
    var acc: f64 = 0.0;
    let stride = num_groups * 256u;
    var i = gid;
    while i < dot_params.len {
        acc += dot_a[i] * dot_b[i];
        i += stride;
    }
    wg_dot[lid] = acc;
    workgroupBarrier();

    // Tree reduction within workgroup
    var offset = 128u;
    while offset > 0u {
        if lid < offset {
            wg_dot[lid] += wg_dot[lid + offset];
        }
        offset >>= 1u;
        workgroupBarrier();
    }

    if lid == 0u {
        dot_result[gid / 256u] = wg_dot[0];
    }
}

// ── Norm2: sqrt(dot(x, x)) — reused with a=b=x ──────────────────
// (No separate shader needed; dot pipeline handles both.)
```

- [ ] **Step 2: Write vector_pipeline.rs**

```rust
// crates/linalg-gpu/src/vector_pipeline.rs
use std::borrow::Cow;
use fem_core::Scalar;
use crate::{GpuContext, GpuVector};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    alpha: f64,
    beta: f64,
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DotParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Pre-compiled vector operations pipelines.
pub struct VectorOpsPipeline {
    axpy_pipeline: wgpu::ComputePipeline,
    axpy_bind_group_layout: wgpu::BindGroupLayout,
    dot_pipeline: wgpu::ComputePipeline,
    dot_bind_group_layout: wgpu::BindGroupLayout,
}

impl VectorOpsPipeline {
    pub fn new(device: &wgpu::Device, native_f64: bool) -> Self {
        let shader_source = if native_f64 {
            Cow::Borrowed(include_str!("vector_ops.wgsl"))
        } else {
            panic!("f64 emulation path not yet implemented");
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("vector_ops_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        // Axpy bind group layout
        let axpy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("axpy_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let axpy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("axpy_layout"),
            bind_group_layouts: &[&axpy_bgl],
            push_constant_ranges: &[],
        });

        let axpy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("axpy_pipeline"),
            layout: Some(&axpy_layout),
            module: &shader,
            entry_point: Some("axpy_main"),
            compilation_options: Default::default(),
        });

        // Dot bind group layout (with result buffer)
        let dot_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let dot_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dot_layout"),
            bind_group_layouts: &[&dot_bgl],
            push_constant_ranges: &[],
        });

        let dot_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dot_pipeline"),
            layout: Some(&dot_layout),
            module: &shader,
            entry_point: Some("dot_main"),
            compilation_options: Default::default(),
        });

        Self {
            axpy_pipeline,
            axpy_bind_group_layout: axpy_bgl,
            dot_pipeline,
            dot_bind_group_layout: dot_bgl,
        }
    }

    /// Encode axpy: `y = alpha * x + beta * y`
    pub fn encode_axpy<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        alpha: f64,
        x: &GpuVector<T>,
        beta: f64,
        y: &GpuVector<T>,
    ) {
        assert_eq!(x.len(), y.len());
        let params = AxpyParams { alpha, beta, len: x.len(), _pad: 0 };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("axpy_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("axpy_bg"),
            layout: &self.axpy_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.buffer().as_entire_binding() },
            ],
        });

        let workgroups = (x.len() + 255) / 256;
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("axpy_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.axpy_pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    /// Encode dot product: result stored in `result_buf` (a GPU buffer of
    /// length n_workgroups, requires a second-pass reduction on CPU).
    ///
    /// The caller should allocate `result_buf` with `n_workgroups * 8` bytes
    /// where `n_workgroups = (len + 255) / 256`.
    pub fn encode_dot<T: Scalar>(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuVector<T>,
        b: &GpuVector<T>,
        result_buf: &wgpu::Buffer,
    ) {
        assert_eq!(a.len(), b.len());
        let n_workgroups = (a.len() + 255u32) / 256u32;
        let params = DotParams { len: a.len(), _pad0: 0, _pad1: 0, _pad2: 0 };
        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("dot_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dot_bg"),
            layout: &self.dot_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: a.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: b.buffer().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: result_buf.as_entire_binding() },
            ],
        });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dot_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.dot_pipeline);
        cpass.set_bind_group(0, &bg, &[]);
        cpass.dispatch_workgroups(n_workgroups, 1, 1);
    }
}
```

- [ ] **Step 3: Update lib.rs**

Edit `crates/linalg-gpu/src/lib.rs`:
```rust
pub use vector_pipeline::VectorOpsPipeline;
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p fem-linalg-gpu 2>&1`
Expected: success

- [ ] **Step 5: Commit**

```bash
git add crates/linalg-gpu/src/vector_ops.wgsl crates/linalg-gpu/src/vector_pipeline.rs crates/linalg-gpu/src/lib.rs
git commit -m "feat: vector ops pipeline (axpy, dot reduction)"
```

---

### Task 9: Vector ops integration test

**Files:**
- Create: `crates/linalg-gpu/tests/vector_ops_test.rs`

- [ ] **Step 1: Write the test**

```rust
// crates/linalg-gpu/tests/vector_ops_test.rs
use fem_linalg_gpu::{GpuContext, GpuVector, VectorOpsPipeline};

fn ctx() -> GpuContext {
    GpuContext::new_sync().expect("gpu context")
}

#[test]
fn axpy_simple() {
    let gpu = ctx();
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let x = GpuVector::from_slice(&gpu, &[1.0f64, 2.0, 3.0]);
    let y = GpuVector::from_slice(&gpu, &[4.0f64, 5.0, 6.0]);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    // y = 2*x + 0.5*y = [2+2, 4+2.5, 6+3] = [4, 6.5, 9]
    pipeline.encode_axpy(&gpu, &mut encoder, 2.0, &x, 0.5, &y);
    gpu.queue.submit(Some(encoder.finish()));

    let result = y.read_to_cpu(&gpu);
    assert!((result[0] - 4.0).abs() < 1e-14);
    assert!((result[1] - 6.5).abs() < 1e-14);
    assert!((result[2] - 9.0).abs() < 1e-14);
}

#[test]
fn dot_simple() {
    let gpu = ctx();
    let pipeline = VectorOpsPipeline::new(&gpu.device, gpu.features.native_f64);

    let a = GpuVector::from_slice(&gpu, &[1.0f64, 2.0, 3.0]);
    let b = GpuVector::from_slice(&gpu, &[4.0f64, 5.0, 6.0]);
    let n_wg = (a.len() + 255) / 256;
    let result_buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_result"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    pipeline.encode_dot(&gpu, &mut encoder, &a, &b, &result_buf);
    gpu.queue.submit(Some(encoder.finish()));

    // Read back the partial reduction result and sum on CPU
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dot_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(&result_buf, 0, &staging, 0, n_wg as u64 * 8);
    gpu.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    gpu.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let partials: &[f64] = bytemuck::cast_slice(&slice.get_mapped_range());
    let dot: f64 = partials.iter().sum();
    drop(slice);
    staging.unmap();

    let expected = 1.0*4.0 + 2.0*5.0 + 3.0*6.0; // = 4+10+18 = 32
    assert!((dot - expected).abs() < 1e-13, "dot={dot} expected={expected}");
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p fem-linalg-gpu --test vector_ops_test -- --test-threads=1 2>&1`
Expected: 2 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/linalg-gpu/tests/vector_ops_test.rs
git commit -m "test: vector ops (axpy, dot) correctness"
```

---

### Task 10: Add gpu feature to fem-solver and connect linalg-gpu

**Files:**
- Modify: `crates/solver/Cargo.toml`
- Modify: `crates/solver/src/lib.rs`

- [ ] **Step 1: Add gpu feature and optional dependency**

Edit `crates/solver/Cargo.toml`:

```toml
[features]
default  = []
gpu      = ["dep:fem-linalg-gpu"]

[dependencies.fem-linalg-gpu]
path = "../linalg-gpu"
optional = true
```

- [ ] **Step 2: Add gpu module gate in lib.rs**

After the existing `pub use linger::Preconditioner as LingerPreconditioner;` line in `crates/solver/src/lib.rs`, add:

```rust
#[cfg(feature = "gpu")]
pub mod cg_gpu;
#[cfg(feature = "gpu")]
pub mod gmres_gpu;
```

- [ ] **Step 3: Verify compilation with and without gpu feature**

Run: `cargo check -p fem-solver 2>&1`
Expected: success (no gpu dependencies linked)

Run: `cargo check -p fem-solver --features gpu 2>&1`
Expected: success (fem-linalg-gpu linked)

- [ ] **Step 4: Commit**

```bash
git add crates/solver/Cargo.toml crates/solver/src/lib.rs
git commit -m "feat: gpu feature gate for fem-solver"
```

---

### Task 11: GPU CG solver

**Files:**
- Create: `crates/solver/src/cg_gpu.rs`

- [ ] **Step 1: Write GPU-resident CG solver**

```rust
// crates/solver/src/cg_gpu.rs
//! GPU-resident Conjugate Gradient solver.
//!
//! All vectors live on the GPU; only the residual norm is read back each
//! iteration for convergence checking.

use fem_core::Scalar;
use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline,
};
use crate::{SolverConfig, SolveResult, SolverError};

/// Solve `A x = b` using Conjugate Gradient, with all iteration data on the GPU.
///
/// # Arguments
/// * `ctx`     — initialized GPU context with device and queue.
/// * `a`       — system matrix (CPU CSR, uploaded once).
/// * `b`       — right-hand side (CPU slice, uploaded once).
/// * `x`       — initial guess on entry (CPU slice), solution on exit (overwritten).
/// * `cfg`     — convergence parameters.
pub fn solve_cg_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows as u32;
    assert_eq!(a.ncols as u32, n, "matrix must be square");
    assert_eq!(b.len() as u32, n);
    assert_eq!(x.len() as u32, n);

    let spmv_pipeline = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
    let vec_pipeline = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);

    // Upload matrix and vectors
    let gpu_a = GpuCsrMatrix::<f64>::from_cpu(ctx, a);
    let gpu_b = GpuVector::from_slice(ctx, b);
    let mut gpu_x = GpuVector::from_slice(ctx, x);
    let mut gpu_r = GpuVector::<f64>::zeros(ctx, n);
    let mut gpu_p = GpuVector::<f64>::zeros(ctx, n);
    let mut gpu_ap = GpuVector::<f64>::zeros(ctx, n);

    // r = b - A*x  (three submits: tmp=Ax, r=b, r=r-tmp)
    let mut gpu_tmp = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &gpu_x, 0.0, &mut gpu_tmp);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_b, 0.0, &mut gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, -1.0, &gpu_tmp, 1.0, &mut gpu_r);
        ctx.queue.submit(Some(enc.finish()));
    }
    // p = r
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_r, 0.0, &mut gpu_p);
        ctx.queue.submit(Some(enc.finish()));
    }

    // rsold = r·r via dot reduction
    let n_wg = (n + 255) / 256;
    let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cg_dot_buf"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut rsold = {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vec_pipeline.encode_dot(ctx, &mut enc, &gpu_r, &gpu_r, &dot_buf);
        ctx.queue.submit(Some(enc.finish()));
        read_partial_reduction(ctx, &dot_buf, n_wg)
    };

    let b_norm = compute_norm2_gpu(ctx, &vec_pipeline, &gpu_b, &dot_buf);
    let tol = cfg.atol.max(cfg.rtol * b_norm);

    for iter in 0..cfg.max_iter {
        // ap = A * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            spmv_pipeline.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &gpu_p, 0.0, &mut gpu_ap);
            ctx.queue.submit(Some(enc.finish()));
        }

        // pAp = p · ap
        let pap = {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_dot(ctx, &mut enc, &gpu_p, &gpu_ap, &dot_buf);
            ctx.queue.submit(Some(enc.finish()));
            read_partial_reduction(ctx, &dot_buf, n_wg)
        };

        let alpha = rsold / pap;

        // x = x + alpha * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, alpha, &gpu_p, 1.0, &mut gpu_x);
            ctx.queue.submit(Some(enc.finish()));
        }

        // r = r - alpha * ap
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_axpy(ctx, &mut enc, -alpha, &gpu_ap, 1.0, &mut gpu_r);
            ctx.queue.submit(Some(enc.finish()));
        }

        // rsnew = r·r
        let rsnew = {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vec_pipeline.encode_dot(ctx, &mut enc, &gpu_r, &gpu_r, &dot_buf);
            ctx.queue.submit(Some(enc.finish()));
            read_partial_reduction(ctx, &dot_buf, n_wg)
        };

        // Convergence check
        let r_norm = rsnew.sqrt();
        if r_norm < tol {
            // Read solution back
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult {
                converged: true,
                iterations: iter + 1,
                final_residual: r_norm,
            });
        }

        let beta = rsnew / rsold;
        rsold = rsnew;

        // p = r + beta * p
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            // First: p = beta * p (scale in place)
            vec_pipeline.encode_axpy(ctx, &mut enc, 0.0, &gpu_p, beta, &mut gpu_p); // p = 0*r + beta*p = beta*p
            ctx.queue.submit(Some(enc.finish()));
        }
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            // p = r + 1.0*p (p currently = beta*p from above)
            vec_pipeline.encode_axpy(ctx, &mut enc, 1.0, &gpu_r, 1.0, &mut gpu_p);
            ctx.queue.submit(Some(enc.finish()));
        }
    }

    // Did not converge
    let cpu_x = gpu_x.read_to_cpu(ctx);
    x.copy_from_slice(&cpu_x);
    let final_r = compute_norm2_gpu(ctx, &vec_pipeline, &gpu_r, &dot_buf);
    Err(SolverError::ConvergenceFailed {
        max_iter: cfg.max_iter,
        residual: final_r,
    })
}

/// Read back a partial reduction result from GPU dot product and sum on CPU.
fn read_partial_reduction(ctx: &GpuContext, result_buf: &wgpu::Buffer, n_wg: u32) -> f64 {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(result_buf, 0, &staging, 0, n_wg as u64 * 8);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let partials: &[f64] = bytemuck::cast_slice(&slice.get_mapped_range());
    let sum: f64 = partials.iter().sum();
    drop(slice);
    staging.unmap();
    sum
}

fn compute_norm2_gpu(
    ctx: &GpuContext,
    vec_pipeline: &VectorOpsPipeline,
    v: &GpuVector<f64>,
    dot_buf: &wgpu::Buffer,
) -> f64 {
    let n_wg = (v.len() + 255) / 256;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    vec_pipeline.encode_dot(ctx, &mut enc, v, v, dot_buf);
    ctx.queue.submit(Some(enc.finish()));
    read_partial_reduction(ctx, dot_buf, n_wg).sqrt()
}
```

- [ ] **Step 2: Verify compilation with gpu feature**

Run: `cargo check -p fem-solver --features gpu 2>&1`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add crates/solver/src/cg_gpu.rs
git commit -m "feat: GPU-resident Conjugate Gradient solver"
```

---

### Task 12: GPU CG integration test

**Files:**
- Create: `crates/solver/tests/cg_gpu_test.rs`

- [ ] **Step 1: Write the integration test**

This test creates a Poisson-like 2D Laplacian (5-point stencil) on a small grid, solves with GPU CG, and verifies against known solution.

```rust
// crates/solver/tests/cg_gpu_test.rs
#![cfg(feature = "gpu")]

use fem_linalg::CsrMatrix;
use fem_linalg_gpu::GpuContext;
use fem_solver::{SolverConfig, SolveResult, cg_gpu::solve_cg_gpu};

/// Build a 1D Poisson matrix (tridiagonal [2, -1, 0, ...; -1, 2, -1, ...]).
fn poisson_1d(n: usize) -> (CsrMatrix<f64>, Vec<f64>, Vec<f64>) {
    let nnz = 3 * n - 2; // 3 per row except first/last have 2
    let mut row_ptr = vec![0usize; n + 1];
    let mut col_idx = Vec::with_capacity(nnz);
    let mut values = Vec::with_capacity(nnz);

    for i in 0..n {
        row_ptr[i + 1] = row_ptr[i];
        if i > 0 {
            col_idx.push(i as u32 - 1);
            values.push(-1.0);
            row_ptr[i + 1] += 1;
        }
        col_idx.push(i as u32);
        values.push(2.0);
        row_ptr[i + 1] += 1;
        if i + 1 < n {
            col_idx.push(i as u32 + 1);
            values.push(-1.0);
            row_ptr[i + 1] += 1;
        }
    }

    let a = CsrMatrix { nrows: n, ncols: n, row_ptr, col_idx, values };

    // Exact solution: x_i = sin(pi * i / (n-1))
    let pi = std::f64::consts::PI;
    let x_exact: Vec<f64> = (0..n).map(|i| (pi * i as f64 / (n as f64 - 1.0)).sin()).collect();

    // RHS b = A * x_exact
    let mut b = vec![0.0f64; n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut s = 0.0;
        for k in start..end {
            s += a.values[k] * x_exact[a.col_idx[k] as usize];
        }
        b[i] = s;
    }

    (a, b, x_exact)
}

#[test]
fn cg_gpu_solves_poisson_1d() {
    let gpu = GpuContext::new_sync().expect("gpu context");
    let n = 64;
    let (a, b, x_exact) = poisson_1d(n);

    let cfg = SolverConfig {
        rtol: 1e-10,
        atol: 0.0,
        max_iter: 200,
        verbose: false,
        print_level: fem_solver::PrintLevel::Silent,
    };

    let mut x = vec![0.0f64; n];
    let result = solve_cg_gpu(&gpu, &a, &b, &mut x, &cfg).expect("CG should converge");

    assert!(result.converged, "CG did not converge in {} iters", result.iterations);
    assert!(result.iterations <= n, "CG took {} iterations (expected ≤ {n})", result.iterations);

    let mut max_err = 0.0f64;
    for i in 0..n {
        let err = (x[i] - x_exact[i]).abs();
        if err > max_err { max_err = err; }
    }
    assert!(max_err < 1e-8, "max error {max_err} > 1e-8");
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p fem-solver --features gpu --test cg_gpu_test -- --test-threads=1 2>&1`
Expected: 1 test pass (cg_gpu_solves_poisson_1d)

- [ ] **Step 3: Commit**

```bash
git add crates/solver/tests/cg_gpu_test.rs
git commit -m "test: GPU CG Poisson 1D convergence test"
```

---

### Task 13: GPU GMRES solver

**Files:**
- Create: `crates/solver/src/gmres_gpu.rs`

- [ ] **Step 1: Write GPU GMRES with restart**

```rust
// crates/solver/src/gmres_gpu.rs
//! GPU-resident restarted GMRES solver.
//!
//! Uses modified Gram-Schmidt for Arnoldi with m-step restart.
//! The Hessenberg least-squares is solved on CPU (tiny, O(m²)).

use fem_linalg::CsrMatrix;
use fem_linalg_gpu::{
    GpuContext, GpuCsrMatrix, GpuVector,
    SpmvPipeline, VectorOpsPipeline,
};
use crate::{SolverConfig, SolveResult, SolverError};

/// Default GMRES restart dimension.
const DEFAULT_RESTART: usize = 30;

/// Solve `A x = b` using restarted GMRES on the GPU.
pub fn solve_gmres_gpu(
    ctx: &GpuContext,
    a: &CsrMatrix<f64>,
    b: &[f64],
    x: &mut [f64],
    cfg: &SolverConfig,
) -> Result<SolveResult, SolverError> {
    let n = a.nrows as u32;
    let restart = DEFAULT_RESTART.min(n as usize);

    let spmv = SpmvPipeline::new(&ctx.device, ctx.features.native_f64);
    let vops = VectorOpsPipeline::new(&ctx.device, ctx.features.native_f64);

    let gpu_a = GpuCsrMatrix::<f64>::from_cpu(ctx, a);
    let gpu_b = GpuVector::from_slice(ctx, b);
    let mut gpu_x = GpuVector::from_slice(ctx, x);

    let b_norm = compute_norm2(ctx, &vops, &gpu_b);
    let tol = cfg.atol.max(cfg.rtol * b_norm);
    let n_wg = (n + 255) / 256;
    let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gmres_dot"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let mut iter_count = 0usize;

    for _outer in 0..cfg.max_iter / restart + 1 {
        // r = b - A*x
        let mut gpu_r = compute_residual(ctx, &spmv, &vops, &gpu_a, &gpu_b, &gpu_x, n);
        let r_norm = compute_norm2(ctx, &vops, &gpu_r);

        if r_norm < tol {
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: r_norm });
        }

        // Arnoldi basis vectors: V[0..restart] each of length n
        let mut basis: Vec<GpuVector<f64>> = Vec::with_capacity(restart + 1);
        // V[0] = r / r_norm
        {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(ctx, &mut enc, 1.0 / r_norm, &gpu_r, 0.0, &mut gpu_r);
            ctx.queue.submit(Some(enc.finish()));
        }
        basis.push(gpu_r); // gpu_r is now the normalized V[0]

        // Hessenberg matrix (CPU, small)
        let mut h = vec![0.0f64; (restart + 1) * restart];
        let mut s = vec![0.0f64; restart + 1]; // for Givens
        let mut cs = vec![0.0f64; restart];
        let mut sn = vec![0.0f64; restart];

        // RHS of the least-squares: s[0] = ||r0||, rest zero
        s[0] = r_norm;

        let mut gmres_r_norm = r_norm;
        let mut j = 0usize;

        for jj in 0..restart {
            j = jj;
            // w = A * V[j]
            let mut gpu_w = GpuVector::<f64>::zeros(ctx, n);
            {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                spmv.encode_spmv(ctx, &mut enc, 1.0, &gpu_a, &basis[jj], 0.0, &mut gpu_w);
                ctx.queue.submit(Some(enc.finish()));
            }

            // Modified Gram-Schmidt
            for i in 0..=jj {
                let dot_val = {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    vops.encode_dot(ctx, &mut enc, &gpu_w, &basis[i], &dot_buf);
                    ctx.queue.submit(Some(enc.finish()));
                    read_partial_reduction(ctx, &dot_buf, n_wg)
                };
                h[i * restart + jj] = dot_val;
                // w = w - h_ij * V[i]
                {
                    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                    vops.encode_axpy(ctx, &mut enc, -dot_val, &basis[i], 1.0, &mut gpu_w);
                    ctx.queue.submit(Some(enc.finish()));
                }
            }

            let w_norm = compute_norm2(ctx, &vops, &gpu_w);
            h[(jj + 1) * restart + jj] = w_norm;

            // Apply previous Givens rotations to the new column of H
            for i in 0..jj {
                let hi = h[i * restart + jj];
                let hi1 = h[(i + 1) * restart + jj];
                h[i * restart + jj]     =  cs[i] * hi + sn[i] * hi1;
                h[(i + 1) * restart + jj] = -sn[i] * hi + cs[i] * hi1;
            }

            // Compute new Givens rotation for the current column
            let h_jj = h[jj * restart + jj];
            let h_j1j = h[(jj + 1) * restart + jj];
            let denom = (h_jj * h_jj + h_j1j * h_j1j).sqrt();
            cs[jj] = h_jj / denom;
            sn[jj] = h_j1j / denom;
            h[jj * restart + jj] = denom;
            h[(jj + 1) * restart + jj] = 0.0;

            // Apply to s
            let sj = s[jj];
            let sj1 = s[jj + 1];
            s[jj]     =  cs[jj] * sj + sn[jj] * sj1;
            s[jj + 1] = -sn[jj] * sj + cs[jj] * sj1;

            gmres_r_norm = s[jj + 1].abs();
            iter_count += 1;

            if gmres_r_norm < tol {
                break;
            }

            // V[j+1] = w / w_norm
            if w_norm > 1e-15 {
                let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
                vops.encode_axpy(ctx, &mut enc, 1.0 / w_norm, &gpu_w, 0.0, &mut gpu_w);
                ctx.queue.submit(Some(enc.finish()));
            }
            basis.push(gpu_w);
        }

        // Back-substitute on CPU to solve H * y = s
        let mut y = vec![0.0f64; j + 1];
        for ii in (0..=j).rev() {
            let mut sum = s[ii];
            for kk in ii + 1..=j {
                sum -= h[ii * restart + kk] * y[kk];
            }
            y[ii] = sum / h[ii * restart + ii];
        }

        // x = x + sum(y[i] * V[i])
        for i in 0..=j {
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            vops.encode_axpy(ctx, &mut enc, y[i], &basis[i], 1.0, &mut gpu_x);
            ctx.queue.submit(Some(enc.finish()));
        }

        if gmres_r_norm < tol {
            let cpu_x = gpu_x.read_to_cpu(ctx);
            x.copy_from_slice(&cpu_x);
            return Ok(SolveResult { converged: true, iterations: iter_count, final_residual: gmres_r_norm });
        }
    }

    let cpu_x = gpu_x.read_to_cpu(ctx);
    x.copy_from_slice(&cpu_x);
    Err(SolverError::ConvergenceFailed { max_iter: cfg.max_iter, residual: 0.0 })
}

fn compute_norm2(
    ctx: &GpuContext,
    vops: &VectorOpsPipeline,
    v: &GpuVector<f64>,
) -> f64 {
    let n_wg = (v.len() + 255) / 256;
    let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tmp_norm"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    vops.encode_dot(ctx, &mut enc, v, v, &dot_buf);
    ctx.queue.submit(Some(enc.finish()));
    read_partial_reduction(ctx, &dot_buf, n_wg).sqrt()
}

fn compute_residual(
    ctx: &GpuContext,
    spmv: &SpmvPipeline,
    vops: &VectorOpsPipeline,
    a: &GpuCsrMatrix<f64>,
    b: &GpuVector<f64>,
    x: &GpuVector<f64>,
    n: u32,
) -> GpuVector<f64> {
    let mut ax = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        spmv.encode_spmv(ctx, &mut enc, 1.0, a, x, 0.0, &mut ax);
        ctx.queue.submit(Some(enc.finish()));
    }
    let mut r = GpuVector::<f64>::zeros(ctx, n);
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, 1.0, b, 0.0, &mut r);
        ctx.queue.submit(Some(enc.finish()));
    }
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        vops.encode_axpy(ctx, &mut enc, -1.0, &ax, 1.0, &mut r);
        ctx.queue.submit(Some(enc.finish()));
    }
    r
}

fn read_partial_reduction(ctx: &GpuContext, result_buf: &wgpu::Buffer, n_wg: u32) -> f64 {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(result_buf, 0, &staging, 0, n_wg as u64 * 8);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let partials: &[f64] = bytemuck::cast_slice(&slice.get_mapped_range());
    let sum: f64 = partials.iter().sum();
    drop(slice);
    staging.unmap();
    sum
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p fem-solver --features gpu 2>&1`
Expected: success

- [ ] **Step 3: Commit**

```bash
git add crates/solver/src/gmres_gpu.rs
git commit -m "feat: GPU-resident GMRES solver with restart"
```

---

### Task 14: End-to-end refactoring and cleanup pass

**Files:**
- Modify: `crates/linalg-gpu/src/lib.rs`
- Modify: `crates/linalg-gpu/src/spmv_pipeline.rs`
- Modify: `crates/linalg-gpu/src/vector_pipeline.rs`
- Modify: `crates/solver/src/cg_gpu.rs`
- Modify: `crates/solver/src/gmres_gpu.rs`

- [ ] **Step 1: Consolidate shared GPU helpers into linalg-gpu**

Move `compute_norm2` and `read_partial_reduction` into `fem-linalg-gpu` so both CG and GMRES can use them without duplication.

Add to `crates/linalg-gpu/src/vector_pipeline.rs`:

```rust
impl VectorOpsPipeline {
    // ...existing code...

    /// Compute ||v||₂ by dispatching dot(v,v) and reading back the
    /// partial reduction. Creates a temporary result buffer internally.
    pub fn compute_norm2<T: Scalar>(&self, ctx: &GpuContext, v: &GpuVector<T>) -> f64 {
        let n_wg = (v.len() + 255) / 256;
        let dot_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("norm2_tmp"),
            size: n_wg as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.encode_dot(ctx, &mut enc, v, v, &dot_buf);
        ctx.queue.submit(Some(enc.finish()));
        read_partial_reduction(ctx, &dot_buf, n_wg).sqrt()
    }
}

/// Read back a GPU partial-reduction buffer (post dot dispatch) and sum on CPU.
pub fn read_partial_reduction(ctx: &GpuContext, result_buf: &wgpu::Buffer, n_wg: u32) -> f64 {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reduce_staging"),
        size: n_wg as u64 * 8,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(result_buf, 0, &staging, 0, n_wg as u64 * 8);
    ctx.queue.submit(Some(enc.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let partials: &[f64] = bytemuck::cast_slice(&slice.get_mapped_range());
    let sum: f64 = partials.iter().sum();
    drop(slice);
    staging.unmap();
    sum
}
```

- [ ] **Step 2: Update cg_gpu.rs to use shared helpers**

Replace the local `read_partial_reduction` and `compute_norm2_gpu` with calls to `vector_pipeline::read_partial_reduction` and `VectorOpsPipeline::compute_norm2`.

- [ ] **Step 3: Update gmres_gpu.rs to use shared helpers**

Same — remove local copies, use the crate-level helpers.

- [ ] **Step 4: Update lib.rs re-exports**

```rust
pub use vector_pipeline::{VectorOpsPipeline, read_partial_reduction};
```

- [ ] **Step 5: Run all GPU tests**

Run: `cargo test -p fem-linalg-gpu -- --test-threads=1 2>&1`
Run: `cargo test -p fem-solver --features gpu -- --test-threads=1 2>&1`
Expected: all tests pass

- [ ] **Step 6: Run CPU tests to verify no regression**

Run: `cargo test -p fem-linalg -p fem-solver -- --test-threads=1 2>&1`
Expected: all existing CPU tests still pass

- [ ] **Step 7: Commit**

```bash
git add crates/linalg-gpu/src/vector_pipeline.rs crates/linalg-gpu/src/lib.rs crates/solver/src/cg_gpu.rs crates/solver/src/gmres_gpu.rs
git commit -m "refactor: consolidate GPU reduction helpers into linalg-gpu"
```

---

## What This Plan Covers

- **P0 complete**: `fem-linalg-gpu` crate, `GpuContext`, `GpuVector`, `GpuCsrMatrix`, SpMV + axpy + dot WGSL shaders, correctness tests
- **P1 complete**: `GpuCgSolver`, `GpuGmresSolver`, GPU-resident iteration, convergence via staging readback
- Feature gate: `gpu` flag in `fem-solver`, opt-in

## What This Plan Does NOT Cover (P2-P3 follow-ups)

- Reed wgpu backend wiring (P2) — needs reed upstream wgpu support verification first
- WASM f64 emulation path (P3) — deferred until desktop path is validated
- GPU AMG — requires coarse-grid transfer, deferred to P3
- Buffer pool for reuse — optimization, early but not critical
- Performance benchmarking vs CPU — needs criterion harness in follow-up
- Jacobi GPU preconditioner — straightforward extension of vector ops, can be added in P1 follow-up
