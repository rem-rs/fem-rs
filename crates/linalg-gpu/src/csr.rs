//! GPU-resident CSR (Compressed Sparse Row) matrix.
//!
//! Stores row pointers, column indices, and values in three
//! `DeviceBuffer`s.  Supports construction from a host-side
//! `fem-core` CSR matrix and SpMV dispatch.

use crate::buffer::DeviceBuffer;

/// A CSR sparse matrix resident on the GPU.
///
/// The three arrays follow the standard CSR layout:
/// - `row_ptr[0..nrows+1]` — offsets into `col_idx`/`values`
/// - `col_idx` — column index per non-zero
/// - `values`  — numeric value per non-zero
pub struct GpuCsrMatrix<T: bytemuck::Pod> {
    _nrows: usize,
    _ncols: usize,
    _nnz: usize,
    _row_ptr: DeviceBuffer<u32>,
    _col_idx: DeviceBuffer<u32>,
    _values: DeviceBuffer<T>,
}
