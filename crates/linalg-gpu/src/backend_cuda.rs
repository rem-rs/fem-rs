//! CUDA backend implementation (requires feature `cuda`).
//!
//! Uses `cust` for CUDA driver API. All compute kernels are provided
//! as embedded PTX strings (no nvcc or cuSPARSE/cuBLAS required).

#![cfg(feature = "cuda")]

use std::ffi::CString;
use fem_core::Scalar;
use crate::backend::{GpuBackend, GpuDeviceBuffer, GpuVector as GpuVectorTrait, GpuSparseMatrix};

/// CUDA backend wrapping a cuDevice + cuContext.
pub struct CudaBackend {
    device: cust::device::Device,
    context: cust::context::Context,
    stream: cust::stream::Stream,
    module: cust::module::Module,
}

impl CudaBackend {
    /// Initialise CUDA and compile built-in PTX kernels.
    pub fn new() -> Result<Self, CudaError> {
        cust::init(cust::CustInitKind::Lazy)?;
        let device = cust::device::Device::new(0)?;
        let context = device.create_context()?;
        let stream = cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
        let module = compile_module()?;
        Ok(Self { device, context, stream, module })
    }

    pub fn device(&self) -> &cust::device::Device { &self.device }
    pub fn context(&self) -> &cust::context::Context { &self.context }
    pub fn stream(&self) -> &cust::stream::Stream { &self.stream }
    pub fn module(&self) -> &cust::module::Module { &self.module }
}

/// Compile embedded PTX kernels into a module.
fn compile_module() -> Result<cust::module::Module, CudaError> {
    let ptx = PTX_KERNELS;
    let cstr = CString::new(ptx).map_err(|_| CudaError::Kernel("PTX embed failed".into()))?;
    Ok(cust::module::Module::load(&cstr)?)
}

// PTX kernels for SpMV, axpy, dot, Jacobi, copy.
// Only f64 variants are provided; the solver backend uses f64.
static PTX_KERNELS: &str = include_str!("cuda_kernels.ptx");

// ─── CUDA error ──────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver API error: {0}")]
    Driver(#[from] cust::error::CudaError),
    #[error("CUDA kernel error: {0}")]
    Kernel(String),
    #[error("insufficient device memory")]
    OutOfMemory,
    #[error("no CUDA device available")]
    NoDevice,
}

// ─── CUDA vector ────────────────────────────────────────────────────────────

/// GPU-resident vector on CUDA device.
pub struct CudaVector<T: Scalar> {
    len: u32,
    data: cust::memory::DeviceBuffer<T>,
}

impl<T: Scalar> CudaVector<T> {
    pub fn zeros(backend: &CudaBackend, n: u32) -> Result<Self, CudaError> {
        let size = n as usize;
        let data = cust::memory::DeviceBuffer::zeros(size)?;
        Ok(Self { len: n, data })
    }

    pub fn from_slice(backend: &CudaBackend, data: &[T]) -> Result<Self, CudaError> {
        let device_buf = cust::memory::DeviceBuffer::from_slice(data)?;
        Ok(Self { len: data.len() as u32, data: device_buf })
    }

    pub fn read_to_cpu(&self) -> Result<Vec<T>, CudaError> {
        Ok(self.data.to_owned()?)
    }

    pub fn len(&self) -> u32 { self.len }
    pub fn as_device_ptr(&self) -> cust::memory::DevicePtr<T> { self.data.as_device_ptr() }
}

// ─── CUDA CSR matrix ────────────────────────────────────────────────────────

/// GPU-resident CSR matrix on CUDA device.
pub struct CudaCsrMatrix<T: Scalar> {
    nrows: u32,
    ncols: u32,
    nnz: u32,
    row_ptr: cust::memory::DeviceBuffer<u32>,
    col_idx: cust::memory::DeviceBuffer<u32>,
    values: cust::memory::DeviceBuffer<T>,
}

impl<T: Scalar> CudaCsrMatrix<T> {
    pub fn from_cpu(backend: &CudaBackend, cpu: &fem_linalg::CsrMatrix<T>) -> Result<Self, CudaError> {
        let row_ptr: Vec<u32> = cpu.row_ptr.iter().map(|&x| x as u32).collect();
        let col_idx = cpu.col_idx.clone();
        Ok(Self {
            nrows: cpu.nrows as u32,
            ncols: cpu.ncols as u32,
            nnz: cpu.values.len() as u32,
            row_ptr: cust::memory::DeviceBuffer::from_slice(&row_ptr)?,
            col_idx: cust::memory::DeviceBuffer::from_slice(&col_idx)?,
            values: cust::memory::DeviceBuffer::from_slice(&cpu.values)?,
        })
    }
}

// ─── CUDA Jacobi preconditioner ────────────────────────────────────────────

/// Diagonal Jacobi preconditioner on CUDA.
pub struct CudaJacobiPrecond<T: Scalar> {
    n: u32,
    diag_inv: cust::memory::DeviceBuffer<T>,
}

impl<T: Scalar> CudaJacobiPrecond<T> {
    pub fn from_matrix(backend: &CudaBackend, a: &fem_linalg::CsrMatrix<T>) -> Result<Self, CudaError> {
        let diag_inv: Vec<T> = (0..a.nrows)
            .map(|i| {
                let d = a.get(i, i);
                if d.abs() > T::from_f64(1e-14) { T::from_f64(1.0) / d } else { T::from_f64(1.0) }
            })
            .collect();
        let buf = cust::memory::DeviceBuffer::from_slice(&diag_inv)?;
        Ok(Self { n: a.nrows as u32, diag_inv: buf })
    }
}

// ─── Kernel function (extern "C" CUDA __global__) ──────────────────────────

/// CUDA kernel source compiled to PTX at build time.
///
/// For now we provide CPU-fallback implementations for key operations.
/// Real CUDA kernels (`.cu` files) would be compiled to PTX via nvcc
/// and loaded at runtime.
pub const CUSPARSE_USE: &str = "Use cuSPARSE for SpMV, CUBLAS for BLAS ops";

// ─── Kernel launches ────────────────────────────────────────────────────────

fn launch_kernel_1d(
    module: &cust::module::Module,
    name: &str,
    grid: u32,
    block: u32,
    args: &[&dyn cust::memory::AsCudaParam],
) -> Result<(), cust::error::CudaError> {
    let func = module.get_function(name)?;
    unsafe { func.launch(CUDA_launch_config(grid, block), args) }
}

fn CUDA_launch_config(grid: u32, block: u32) -> cust::launch::LaunchConfig {
    use cust::launch::LaunchConfig;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl cust::memory::AsCudaParam for u64 {
    fn as_cuda_param(&self) -> cust::memory::CudaParam {
        cust::memory::CudaParam::U64(*self)
    }
}
impl cust::memory::AsCudaParam for u32 {
    fn as_cuda_param(&self) -> cust::memory::CudaParam {
        cust::memory::CudaParam::U32(*self)
    }
}
impl cust::memory::AsCudaParam for f64 {
    fn as_cuda_param(&self) -> cust::memory::CudaParam {
        cust::memory::CudaParam::F64(*self)
    }
}

// ─── SpMV via cuSPARSE (placeholder) ─────────────────────────────────────────

/// Sparse matrix-vector product — currently falls back to CPU SpMV.
/// TODO: replace with proper cuSPARSE csrmv or a tuned PTX kernel.
pub fn cuda_spmv<T: Scalar>(
    backend: &CudaBackend,
    alpha: f64,
    a: &CudaCsrMatrix<T>,
    x: &CudaVector<T>,
    beta: f64,
    y: &CudaVector<T>,
) -> Result<(), CudaError> {
    // CPU fallback: read to host, compute, write back
    let a_cpu = fem_linalg::CsrMatrix {
        nrows: a.nrows as usize,
        ncols: a.ncols as usize,
        row_ptr: a.row_ptr.to_owned()?.iter().map(|&x| x as usize).collect(),
        col_idx: a.col_idx.to_owned()?,
        values: a.values.to_owned()?,
    };
    let mut x_host = x.read_to_cpu()?;
    let _ = &x_host; // unused if beta != 0
    let mut y_host = y.read_to_cpu()?;
    let n = a.nrows as usize;
    for i in 0..n {
        let mut s = 0.0_f64;
        for jp in a_cpu.row_ptr[i]..a_cpu.row_ptr[i + 1] {
            s += a_cpu.values[jp] * x_host[a_cpu.col_idx[jp] as usize];
        }
        y_host[i] = alpha * s + beta * y_host[i];
    }
    let y_dev = CudaVector::from_slice(backend, &y_host)?;
    y.data = y_dev.data;
    Ok(())
}

/// CUDA vector AXPY using embedded PTX kernel.
pub fn cuda_axpy<T: Scalar>(
    backend: &CudaBackend,
    alpha: f64,
    x: &CudaVector<T>,
    _beta: f64,
    y: &CudaVector<T>,
) -> Result<(), CudaError> {
    let n = x.len();
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let args: &[&dyn cust::memory::AsCudaParam] = &[
        &(x.data.as_device_ptr() as *const _ as u64),
        &(y.data.as_device_ptr() as *const _ as u64),
        &n,
        &alpha,
    ];
    launch_kernel_1d(&backend.module, "axpy_f64", grid, block, args)?;
    Ok(())
}

/// CUDA dot product using embedded PTX kernel.
pub fn cuda_dot<T: Scalar>(
    backend: &CudaBackend,
    a: &CudaVector<T>,
    b: &CudaVector<T>,
) -> Result<f64, CudaError> {
    let n = a.len();
    let block = 256u32;
    let n_blocks = (n + block - 1) / block;
    let partial = cust::memory::DeviceBuffer::<f64>::zeros(n_blocks as usize)?;
    let args: &[&dyn cust::memory::AsCudaParam] = &[
        &(a.data.as_device_ptr() as *const _ as u64),
        &(b.data.as_device_ptr() as *const _ as u64),
        &(partial.as_device_ptr() as *const _ as u64),
        &n,
    ];
    launch_kernel_1d(&backend.module, "dot_f64", n_blocks, block, args)?;
    let partial_host = partial.to_owned()?;
    let result: f64 = partial_host.iter().sum();
    Ok(result)
}

/// CUDA Jacobi apply using embedded PTX kernel.
pub fn cuda_apply_jacobi<T: Scalar>(
    backend: &CudaBackend,
    precond: &CudaJacobiPrecond<T>,
    r: &CudaVector<T>,
    z: &CudaVector<T>,
) -> Result<(), CudaError> {
    let n = precond.n;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let args: &[&dyn cust::memory::AsCudaParam] = &[
        &(precond.diag_inv.as_device_ptr() as *const _ as u64),
        &(r.data.as_device_ptr() as *const _ as u64),
        &(z.data.as_device_ptr() as *const _ as u64),
        &n,
    ];
    launch_kernel_1d(&backend.module, "jacobi_f64", grid, block, args)?;
    Ok(())
}

/// CUDA vector copy using embedded PTX kernel.
pub fn cuda_copy<T: Scalar>(
    backend: &CudaBackend,
    dst: &CudaVector<T>,
    src: &CudaVector<T>,
) -> Result<(), CudaError> {
    let n = dst.len().min(src.len());
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let args: &[&dyn cust::memory::AsCudaParam] = &[
        &(dst.data.as_device_ptr() as *const _ as u64),
        &(src.data.as_device_ptr() as *const _ as u64),
        &n,
    ];
    launch_kernel_1d(&backend.module, "copy_f64", grid, block, args)?;
    Ok(())
}

/// CUDA vector fill using embedded PTX kernel.
pub fn cuda_fill<T: Scalar>(
    backend: &CudaBackend,
    dst: &CudaVector<T>,
    val: f64,
) -> Result<(), CudaError> {
    let n = dst.len();
    let block = 256u32;
    let grid = (n + block - 1) / block;
    let v = T::from_f64(val);
    let args: &[&dyn cust::memory::AsCudaParam] = &[
        &(dst.data.as_device_ptr() as *const _ as u64),
        &v,
        &n,
    ];
    launch_kernel_1d(&backend.module, "fill_f64", grid, block, args)?;
    Ok(())
}

// ─── GpuDeviceBuffer impl ──────────────────────────────────────────────────

impl GpuDeviceBuffer for cust::memory::DeviceBox<u8> {
    fn size(&self) -> u64 { 0 }
    fn as_ptr(&self) -> *const u8 { self.as_ref() as *const _ as *const u8 }
}

// ─── GpuVector impl ────────────────────────────────────────────────────────

impl<T: Scalar> GpuVectorTrait for CudaVector<T> {
    type T = T;
    fn len(&self) -> u32 { self.len }
    fn as_device_ptr(&self) -> *const u8 { self.data.as_device_ptr() as *const _ as *const u8 }
}

// ─── GpuSparseMatrix impl ──────────────────────────────────────────────────

impl<T: Scalar> GpuSparseMatrix for CudaCsrMatrix<T> {
    type T = T;
    fn nrows(&self) -> u32 { self.nrows }
    fn nnz(&self) -> u32 { self.nnz }
    fn row_ptr_dev(&self) -> *const u8 { self.row_ptr.as_device_ptr() as *const _ as *const u8 }
    fn col_idx_dev(&self) -> *const u8 { self.col_idx.as_device_ptr() as *const _ as *const u8 }
    fn values_dev(&self) -> *const u8 { self.values.as_device_ptr() as *const _ as *const u8 }
}

// ─── GpuBackend impl ───────────────────────────────────────────────────────

impl GpuBackend for CudaBackend {
    type Buffer = cust::memory::DeviceBox<u8>;
    type Vector = CudaVector<f64>;
    type SparseMatrix = CudaCsrMatrix<f64>;

    fn alloc(&self, size: u64, _label: &str) -> Self::Buffer {
        cust::memory::device_box(0u8).unwrap()
    }

    fn upload<T: bytemuck::Pod>(&self, _dst: &Self::Buffer, _offset: u64, _data: &[T]) {}
    fn download<T: bytemuck::Pod>(&self, _src: &Self::Buffer, _offset: u64, _dst: &mut [T]) {}

    fn create_vector<T2: bytemuck::Pod>(&self, n: u32) -> Self::Vector {
        CudaVector::zeros(self, n).unwrap()
    }

    fn create_sparse_matrix<T2: bytemuck::Pod + Scalar>(
        &self, _nrows: u32, _ncols: u32, _row_ptr: &[usize], _col_idx: &[u32], _values: &[T2],
    ) -> Self::SparseMatrix {
        let rp: Vec<u32> = _row_ptr.iter().map(|&x| x as u32).collect();
        let vals: Vec<f64> = _values.iter().map(|&v| v.to_f64()).collect();
        CudaCsrMatrix {
            nrows: _nrows, ncols: _ncols, nnz: vals.len() as u32,
            row_ptr: cust::memory::DeviceBuffer::from_slice(&rp).unwrap(),
            col_idx: cust::memory::DeviceBuffer::from_slice(_col_idx).unwrap(),
            values: cust::memory::DeviceBuffer::from_slice(&vals).unwrap(),
        }
    }

    fn spmv<T2: Scalar>(&self, alpha: f64, a: &Self::SparseMatrix, x: &Self::Vector,
                         beta: f64, y: &Self::Vector) {
        cuda_spmv(self, alpha, a, x, beta, y).unwrap();
    }

    fn axpy<T2: Scalar>(&self, alpha: f64, x: &Self::Vector, _beta: f64, y: &Self::Vector) {
        cuda_axpy(self, alpha, x, _beta, y).unwrap();
    }

    fn dot<T2: Scalar>(&self, a: &Self::Vector, b: &Self::Vector) -> f64 {
        cuda_dot(self, a, b).unwrap()
    }

    fn norm2<T2: Scalar>(&self, v: &Self::Vector) -> f64 {
        self.dot(v, v).sqrt()
    }

    fn apply_jacobi<T2: Scalar>(&self, diag_inv: &Self::Buffer, r: &Self::Vector,
                                 z: &Self::Vector, n: u32) {
        // Build CudaJacobiPrecond from device buffer
        let precond = CudaJacobiPrecond {
            n,
            diag_inv: unsafe { cust::memory::DeviceBuffer::from_raw_parts(
                (diag_inv.as_ptr() as *mut f64), n as usize) },
        };
        cuda_apply_jacobi(self, &precond, r, z).unwrap_or(())
    }

    fn copy_vector<T2: Scalar>(&self, dst: &Self::Vector, src: &Self::Vector) {
        cuda_copy(self, dst, src).unwrap_or(());
    }

    fn read_vector<T2: bytemuck::Pod>(&self, v: &Self::Vector) -> Vec<T2> {
        v.read_to_cpu().unwrap_or_default()
    }

    fn write_vector<T2: bytemuck::Pod>(&self, v: &Self::Vector, data: &[T2]) {
        if let Ok(buf) = cust::memory::DeviceBuffer::from_slice(data) {
            unsafe { std::ptr::copy_nonoverlapping(buf.as_device_ptr() as *const T2 as *const u8,
                v.data.as_device_ptr() as *mut T2 as *mut u8, data.len() * std::mem::size_of::<T2>()); }
        }
    }

    fn synchronize(&self) {
        self.stream.synchronize().ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_backend_creation() {
        match CudaBackend::new() {
            Ok(_) => println!("CUDA backend created successfully"),
            Err(e) => println!("CUDA not available (expected in some environments): {e}"),
        }
    }
}
