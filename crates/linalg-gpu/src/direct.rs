//! GPU-resident direct solvers.
//!
//! For small-to-medium systems: dense LU factorization by copying data
//! between GPU and CPU. The CPU uses BLAS-level optimised LU from
//! `fem-linalg`; data transfer across PCIe is the dominant cost for
//! small systems but remains negligible compared to iterative solver
//! setup overhead.
//!
//! Feature-gated CUDA path (requires `cuda` feature):
//! - `cusparse` for sparse CSR → dense conversion
//! - `cusolver` for GPU-native sparse/dense LU

use crate::{DeviceBuffer, GpuContext, GpuVector};

/// Solve `A x = b` via dense LU, keeping data on GPU.
///
/// `a` is an n×n dense matrix (row-major) uploaded to `DeviceBuffer`.
/// `x` and `b` share the same `GpuVector` — on input it holds the RHS,
/// on output it is overwritten with the solution.
///
/// Internally copies A and b to CPU, factors/solves, and copies the
/// solution back to the GPU vector.  This round-trip is O(n²) in
/// transfers and O(n³) in compute.  For n ≤ 512 it is still faster
/// than thousands of CG iterations.
pub fn solve_dense_gpu(
    ctx: &GpuContext,
    a: &DeviceBuffer,
    b: &mut GpuVector<f64>,
    n: u32,
) -> Result<(), String> {
    assert_eq!(b.len(), n);
    let n_usize = n as usize;

    // Read A from GPU
    let a_size = (n_usize * n_usize) as u64 * 8;
    let a_staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("direct_a_staging"),
        size: a_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_buffer(a.buffer(), 0, &a_staging, 0, a_size);
    ctx.queue.submit(Some(enc.finish()));

    let slice = a_staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let mut a_host: Vec<f64> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    let _ = slice;
    a_staging.unmap();

    // Read b from GPU
    let mut x: Vec<f64> = b.read_to_cpu(ctx);

    // Factor and solve on CPU
    let mut piv = vec![0usize; n_usize];
    fem_linalg::dense::lu_factor(&mut a_host, n_usize, &mut piv)
        .map_err(|e| format!("LU factor failed: {e}"))?;
    fem_linalg::dense::lu_solve(&a_host, n_usize, &piv, &mut x);

    // Write solution back to GPU
    b.write_from_slice(ctx, &x);
    Ok(())
}

/// GPU-resident dense matrix (row-major) for direct solves.
pub struct GpuDenseMatrix {
    n: u32,
    buf: DeviceBuffer,
}

impl GpuDenseMatrix {
    /// Upload an n×n row-major dense matrix to GPU.
    pub fn from_slice(ctx: &GpuContext, data: &[f64], n: u32) -> Self {
        assert_eq!(data.len() as u32, n * n);
        let buf = DeviceBuffer::with_staging(
            &ctx.device,
            (n * n) as u64 * 8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "dense_mat",
        );
        ctx.queue.write_buffer(buf.buffer(), 0, bytemuck::cast_slice(data));
        Self { n, buf }
    }

    pub fn buffer(&self) -> &wgpu::Buffer { self.buf.buffer() }
    pub fn n(&self) -> u32 { self.n }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuContext;

    fn ctx() -> GpuContext {
        GpuContext::new_sync().expect("gpu context")
    }

    #[test]
    fn dense_solve_3x3() {
        // [4, -1, 0; -1, 4, -1; 0, -1, 4] * x = [3; 2; 3];  x = [1; 1; 1]
        let gpu = ctx();
        let n = 3u32;
        let a_data: Vec<f64> = vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0];
        let a_buf = DeviceBuffer::with_staging(
            &gpu.device,
            (n * n) as u64 * 8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "test_dense_a",
        );
        gpu.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(&a_data));

        let b_vec: Vec<f64> = vec![3.0, 2.0, 3.0];
        let mut b_gpu = GpuVector::from_slice(&gpu, &b_vec);

        solve_dense_gpu(&gpu, &a_buf, &mut b_gpu, n).expect("dense solve");
        let x = b_gpu.read_to_cpu(&gpu);
        for &xi in &x {
            assert!((xi - 1.0).abs() < 1e-12, "expected 1.0, got {xi}");
        }
    }

    #[test]
    fn dense_solve_identity() {
        let gpu = ctx();
        let n = 5u32;
        let mut a_data = vec![0.0f64; (n * n) as usize];
        for i in 0..n as usize {
            a_data[i * n as usize + i] = 1.0;
        }
        let a_buf = DeviceBuffer::with_staging(
            &gpu.device,
            (n * n) as u64 * 8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "test_id_a",
        );
        gpu.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(&a_data));

        let rhs: Vec<f64> = (0..n).map(|i| i as f64 * 2.0).collect();
        let mut b_gpu = GpuVector::from_slice(&gpu, &rhs);
        solve_dense_gpu(&gpu, &a_buf, &mut b_gpu, n).expect("id solve");
        let x = b_gpu.read_to_cpu(&gpu);
        for (xi, expected) in x.iter().zip(rhs.iter()) {
            assert!((xi - expected).abs() < 1e-12, "expected {expected}, got {xi}");
        }
    }

    #[test]
    fn gpu_dense_solve_matches_cpu_poisson_2d() {
        // Build a 2D Poisson-like SPD matrix (57 × 57 for 8×8 grid P1).
        // Solve on both CPU and GPU, verify relative difference < 1e-10.
        use fem_linalg::dense::{lu_factor, lu_solve};

        // Build 2D Laplacian on 8×8 grid (57 DOFs after boundary elimination)
        let nx: usize = 8; let ny: usize = 8;
        let n = (nx - 1) * (ny - 1);
        let mut a = vec![0.0_f64; n * n];
        let rhs = vec![1.0_f64; n];

        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let row = j * (nx - 1) + i;
                a[row * n + row] = 4.0;
                if i > 0 { a[row * n + row - 1] = -1.0; }
                if i < nx - 2 { a[row * n + row + 1] = -1.0; }
                if j > 0 { a[row * n + row - (nx - 1)] = -1.0; }
                if j < ny - 2 { a[row * n + row + (nx - 1)] = -1.0; }
            }
        }

        // CPU solve
        let mut cpu_a = a.clone();
        let mut cpu_x = rhs.clone();
        let mut piv = vec![0usize; n];
        lu_factor(&mut cpu_a, n, &mut piv).expect("LU factor");
        lu_solve(&cpu_a, n, &piv, &mut cpu_x);

        // GPU solve
        let gpu = ctx();
        let a_buf = DeviceBuffer::with_staging(
            &gpu.device, (n * n) as u64 * 8,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "mms_a",
        );
        gpu.queue.write_buffer(a_buf.buffer(), 0, bytemuck::cast_slice(&a));
        let mut b_gpu = GpuVector::from_slice(&gpu, &rhs);
        solve_dense_gpu(&gpu, &a_buf, &mut b_gpu, n as u32).expect("GPU dense solve");
        let gpu_x = b_gpu.read_to_cpu(&gpu);

        // Compare
        let max_diff: f64 = cpu_x.iter().zip(gpu_x.iter())
            .map(|(c, g)| (c - g).abs()).reduce(f64::max).unwrap_or(0.0);
        let norm: f64 = cpu_x.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
        let rel_diff = max_diff / norm;
        assert!(rel_diff < 1e-10, "GPU vs CPU relative diff = {:.3e} (max_diff={:.3e})", rel_diff, max_diff);
    }
}
