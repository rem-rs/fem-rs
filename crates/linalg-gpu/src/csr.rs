//! GPU-resident CSR (Compressed Sparse Row) matrix.
//!
//! Stores row pointers, column indices, and values in three
//! `DeviceBuffer`s.  Supports construction from a host-side
//! `fem_linalg::CsrMatrix` and SpMV dispatch.

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
