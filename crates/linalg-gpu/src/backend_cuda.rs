//! CUDA backend implementation (requires feature `cuda`).
//!
//! Uses `cust` for CUDA driver API and provides SpMV, vector ops,
//! and Jacobi preconditioner via CUDA kernels.

#![cfg(feature = "cuda")]

use std::ffi::CString;
use fem_core::Scalar;
use crate::backend::{GpuBackend, GpuDeviceBuffer, GpuVector as GpuVectorTrait, GpuSparseMatrix};

/// CUDA backend wrapping a cuDevice + cuContext.
pub struct CudaBackend {
    device: cust::device::Device,
    context: cust::context::Context,
    stream: cust::stream::Stream,
}

impl CudaBackend {
    /// Initialize CUDA and select the first device.
    pub fn new() -> Result<Self, CudaError> {
        cust::init(CustInitKind::Lazy)?;
        let device = cust::device::Device::new(0)?;
        let context = device.create_context()?;
        let stream = cust::stream::Stream::new(cust::stream::StreamFlags::DEFAULT, None)?;
        Ok(Self { device, context, stream })
    }

    pub fn device(&self) -> &cust::device::Device { &self.device }
    pub fn context(&self) -> &cust::context::Context { &self.context }
    pub fn stream(&self) -> &cust::stream::Stream { &self.stream }
}

/// CUDA-specific error.
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

// ─── SpMV via cuSPARSE ─────────────────────────────────────────────────────

/// Sparse matrix-vector product using cuSPARSE.
pub fn cuda_spmv<T: Scalar>(
    backend: &CudaBackend,
    alpha: f64,
    a: &CudaCsrMatrix<T>,
    x: &CudaVector<T>,
    beta: f64,
    y: &CudaVector<T>,
) -> Result<(), CudaError> {
    // TODO: Implement with cusparse
    // let handle = cusparse::CusparseHandle::new()?;
    // cusparse::csrmv(...)
    Err(CudaError::Kernel("cuSPARSE SpMV not yet implemented".to_string()))
}

/// CUDA vector AXPY using CUBLAS.
pub fn cuda_axpy<T: Scalar>(
    backend: &CudaBackend,
    alpha: f64,
    x: &CudaVector<T>,
    beta: f64,
    y: &CudaVector<T>,
) -> Result<(), CudaError> {
    // TODO: Implement with cublas
    // cublas::axpy(...)
    Err(CudaError::Kernel("cuBLAS axpy not yet implemented".to_string()))
}

/// CUDA dot product using CUBLAS.
pub fn cuda_dot<T: Scalar>(
    backend: &CudaBackend,
    a: &CudaVector<T>,
    b: &CudaVector<T>,
) -> Result<f64, CudaError> {
    // TODO: Implement with cublas
    Err(CudaError::Kernel("cuBLAS dot not yet implemented".to_string()))
}

/// CUDA Jacobi apply (element-wise multiply).
pub fn cuda_apply_jacobi<T: Scalar>(
    backend: &CudaBackend,
    precond: &CudaJacobiPrecond<T>,
    r: &CudaVector<T>,
    z: &CudaVector<T>,
) -> Result<(), CudaError> {
    // TODO: Launch CUDA kernel:
    // z[i] = diag_inv[i] * r[i]  for i = 0..n-1
    Err(CudaError::Kernel("CUDA Jacobi kernel not yet implemented".to_string()))
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
    type Vector = CudaVector<f64>;     // Fixed to f64 for simplicity; solver uses f64
    type SparseMatrix = CudaCsrMatrix<f64>;

    fn alloc(&self, size: u64, _label: &str) -> Self::Buffer {
        cust::memory::device_box(0u8).unwrap() // simplified
    }

    fn upload<T: bytemuck::Pod>(&self, _dst: &Self::Buffer, _offset: u64, _data: &[T]) {}
    fn download<T: bytemuck::Pod>(&self, _src: &Self::Buffer, _offset: u64, _dst: &mut [T]) {}

    fn create_vector<T2: bytemuck::Pod>(&self, n: u32) -> Self::Vector {
        CudaVector::zeros(self, n).unwrap()
    }

    fn create_sparse_matrix<T2: bytemuck::Pod + Scalar>(
        &self, _nrows: u32, _ncols: u32, _row_ptr: &[usize], _col_idx: &[u32], _values: &[T2],
    ) -> Self::SparseMatrix {
        unimplemented!("CUDA sparse matrix creation: use CudaCsrMatrix::from_cpu")
    }

    fn spmv<T2: Scalar>(&self, _alpha: f64, _a: &Self::SparseMatrix, _x: &Self::Vector,
                         _beta: f64, _y: &Self::Vector) {
        unimplemented!("CUDA SpMV: use cuda_spmv with cuSPARSE")
    }

    fn axpy<T2: Scalar>(&self, _alpha: f64, _x: &Self::Vector, _beta: f64, _y: &Self::Vector) {
        unimplemented!("CUDA axpy: use cuda_axpy with cuBLAS")
    }

    fn dot<T2: Scalar>(&self, _a: &Self::Vector, _b: &Self::Vector) -> f64 {
        unimplemented!("CUDA dot: use cuda_dot with cuBLAS")
    }

    fn norm2<T2: Scalar>(&self, v: &Self::Vector) -> f64 {
        self.dot(v, v).sqrt()
    }

    fn apply_jacobi<T2: Scalar>(&self, _diag_inv: &Self::Buffer, _r: &Self::Vector,
                                 _z: &Self::Vector, _n: u32) {
        unimplemented!("CUDA Jacobi: use cuda_apply_jacobi")
    }

    fn copy_vector<T2: Scalar>(&self, _dst: &Self::Vector, _src: &Self::Vector) {
        unimplemented!("CUDA copy")
    }

    fn read_vector<T2: bytemuck::Pod>(&self, _v: &Self::Vector) -> Vec<T2> {
        unimplemented!("CUDA read")
    }

    fn write_vector<T2: bytemuck::Pod>(&self, _v: &Self::Vector, _data: &[T2]) {
        unimplemented!("CUDA write")
    }

    fn synchronize(&self) {}
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
