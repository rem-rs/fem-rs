//! GPU backend abstraction layer.
//!
//! Defines traits for backend-agnostic GPU operations.
//! Implementations: [`WgpuBackend`] (default), [`CudaBackend`] (feature `cuda`).

// ─── Device memory abstraction ──────────────────────────────────────────────

/// A contiguous buffer of bytes on the GPU device.
pub trait GpuDeviceBuffer: Send + Sync {
    fn size(&self) -> u64;
    fn as_ptr(&self) -> *const u8;
}

// ─── Vector abstraction ─────────────────────────────────────────────────────

/// A dense vector on the GPU.
#[allow(clippy::len_without_is_empty)]
pub trait GpuVector: Send + Sync {
    type T: 'static;
    fn len(&self) -> u32;
    fn as_device_ptr(&self) -> *const u8;
}

// ─── SpMV abstraction ──────────────────────────────────────────────────────

/// Compressed Sparse Row matrix on the GPU.
pub trait GpuSparseMatrix: Send + Sync {
    type T: 'static;
    fn nrows(&self) -> u32;
    fn nnz(&self) -> u32;
    fn row_ptr_dev(&self) -> *const u8;
    fn col_idx_dev(&self) -> *const u8;
    fn values_dev(&self) -> *const u8;
}

// ─── Backend trait ──────────────────────────────────────────────────────────

/// Core GPU backend operations.
pub trait GpuBackend: Send + Sync {
    type Buffer: GpuDeviceBuffer;
    type Vector: GpuVector;
    type SparseMatrix: GpuSparseMatrix;

    /// Allocate a device buffer of `size` bytes.
    fn alloc(&self, size: u64, label: &str) -> Self::Buffer;

    /// Upload `data` to device buffer at `offset`.
    fn upload<T: bytemuck::Pod>(&self, dst: &Self::Buffer, offset: u64, data: &[T]);

    /// Download from device buffer into `dst` (blocking).
    fn download<T: bytemuck::Pod>(&self, src: &Self::Buffer, offset: u64, dst: &mut [T]);

    /// Create a zero-initialized vector of length `n`.
    fn create_vector<T: bytemuck::Pod>(&self, n: u32) -> Self::Vector;

    /// Create a sparse matrix from CPU CSR data.
    fn create_sparse_matrix<T: bytemuck::Pod + fem_core::Scalar>(
        &self, nrows: u32, ncols: u32, row_ptr: &[usize], col_idx: &[u32], values: &[T],
    ) -> Self::SparseMatrix;

    /// SpMV: y = alpha * A * x + beta * y
    fn spmv<T: fem_core::Scalar>(
        &self, alpha: f64, a: &Self::SparseMatrix, x: &Self::Vector,
        beta: f64, y: &Self::Vector,
    );

    /// AXPY: y = alpha * x + beta * y
    fn axpy<T: fem_core::Scalar>(
        &self, alpha: f64, x: &Self::Vector, beta: f64, y: &Self::Vector,
    );

    /// Dot product: returns sum_i a_i * b_i
    fn dot<T: fem_core::Scalar>(&self, a: &Self::Vector, b: &Self::Vector) -> f64;

    /// Norm2: sqrt(dot(v, v)). Implementations should call `self.dot(v, v).sqrt()`.
    fn norm2<T: fem_core::Scalar>(&self, v: &Self::Vector) -> f64;

    /// Apply Jacobi preconditioner: z_i = diag_inv[i] * r_i
    fn apply_jacobi<T: fem_core::Scalar>(
        &self, diag_inv: &Self::Buffer, r: &Self::Vector, z: &Self::Vector, n: u32,
    );

    /// Copy vector contents: dst = src
    fn copy_vector<T: fem_core::Scalar>(&self, dst: &Self::Vector, src: &Self::Vector);

    /// Read vector back to CPU (blocking).
    fn read_vector<T: bytemuck::Pod>(&self, v: &Self::Vector) -> Vec<T>;

    /// Write to vector from CPU data (non-blocking).
    fn write_vector<T: bytemuck::Pod>(&self, v: &Self::Vector, data: &[T]);

    /// Synchronize (wait for all pending operations to complete).
    fn synchronize(&self);
}

/// Compile-time backend selection.
#[cfg(not(feature = "cuda"))]
pub type DefaultBackend = crate::backend_wgpu::WgpuBackend;
#[cfg(feature = "cuda")]
pub type DefaultBackend = crate::backend_cuda::CudaBackend;
