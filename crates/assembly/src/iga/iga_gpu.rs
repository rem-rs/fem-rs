//! GPU-accelerated IGA Bezier assembly — high-level bridge.
//!
//! Takes NURBS mesh data, builds per-element GPU input buffers, dispatches
//! WGSL compute shaders, and converts the resulting COO triplets to CSR format.
//!
//! ## Usage
//! ```ignore
//! #[cfg(feature = "gpu")]
//! let k = assemble_iga_diffusion_2d_bezier_gpu(&gpu, &mesh, 1.0, 4);
//! ```
//!
//! ## Supported degrees
//! - 2-D: `p,q ≤ 3` (uniform knot vectors, C = I)
//! - 3-D: `p,q,r ≤ 2` (uniform knot vectors, C = I)

use fem_element::iga::{NurbsMesh2D, NurbsMesh3D, NurbsPatch2DData, NurbsPatch3DData};
use fem_element::bezier_extraction::{self, BezierExtraction2D, BezierExtraction3D};
use fem_linalg::{CooMatrix, CsrMatrix};

use super::iga_utils::nonempty_spans;

// ─── 2-D helpers (GPU-gated) ─────────────────────────────────────────────────

#[cfg(feature = "gpu")]
/// Build per-element GPU input for one 2-D patch.
fn build_gpu_elems_2d(
    pd: &NurbsPatch2DData,
    ext: &BezierExtraction2D,
    dof_offset: usize,
) -> Vec<fem_linalg_gpu::GpuIgaBezier2DElement> {
    let p = ext.degree_u;
    let q = ext.degree_v;
    let np1 = p + 1;
    let nq1 = q + 1;
    let nu = pd.kv_u.n_basis();

    let spans_u = nonempty_spans(&pd.kv_u.knots);
    let spans_v = nonempty_spans(&pd.kv_v.knots);

    let mut elems = Vec::with_capacity(spans_u.len() * spans_v.len());

    for (_, (span_u, _, _)) in spans_u.iter().enumerate() {
        let active_u_base = span_u - p;
        for (_, (span_v, _, _)) in spans_v.iter().enumerate() {
            let active_v_base = span_v - q;

            let mut cpts = [0.0_f64; 32];
            let mut weights = [0.0_f64; 16];
            let mut dofs = [0u32; 16];

            for j in 0..=q {
                for i in 0..=p {
                    let local = j * np1 + i;
                    let patch_idx = (active_v_base + j) * nu + (active_u_base + i);
                    cpts[2 * local] = pd.control_pts[patch_idx][0];
                    cpts[2 * local + 1] = pd.control_pts[patch_idx][1];
                    weights[local] = pd.weights[patch_idx];
                    dofs[local] = (dof_offset + patch_idx) as u32;
                }
            }

            elems.push(fem_linalg_gpu::GpuIgaBezier2DElement {
                cpts,
                weights,
                dofs,
            });
        }
    }

    elems
}

#[cfg(feature = "gpu")]
/// Build per-element GPU input for one 3-D patch.
fn build_gpu_elems_3d(
    pd: &NurbsPatch3DData,
    ext: &BezierExtraction3D,
    dof_offset: usize,
) -> Vec<fem_linalg_gpu::GpuIgaBezier3DElement> {
    let p = ext.degree_u;
    let q = ext.degree_v;
    let r = ext.degree_w;
    let np1 = p + 1;
    let nq1 = q + 1;
    let nr1 = r + 1;
    let nu = pd.kv_u.n_basis();
    let nv = pd.kv_v.n_basis();

    let spans_u = nonempty_spans(&pd.kv_u.knots);
    let spans_v = nonempty_spans(&pd.kv_v.knots);
    let spans_w = nonempty_spans(&pd.kv_w.knots);

    let mut elems = Vec::with_capacity(spans_u.len() * spans_v.len() * spans_w.len());

    for (_, (span_u, _, _)) in spans_u.iter().enumerate() {
        let active_u_base = span_u - p;
        for (_, (span_v, _, _)) in spans_v.iter().enumerate() {
            let active_v_base = span_v - q;
            for (_, (span_w, _, _)) in spans_w.iter().enumerate() {
                let active_w_base = span_w - r;

                let mut cpts = [0.0_f64; 81];
                let mut weights = [0.0_f64; 27];
                let mut dofs = [0u32; 27];

                for k in 0..=r {
                    for j in 0..=q {
                        for i in 0..=p {
                            let local = k * nq1 * np1 + j * np1 + i;
                            let patch_idx = (active_w_base + k) * nu * nv
                                + (active_v_base + j) * nu
                                + (active_u_base + i);
                            cpts[3 * local] = pd.control_pts[patch_idx][0];
                            cpts[3 * local + 1] = pd.control_pts[patch_idx][1];
                            cpts[3 * local + 2] = pd.control_pts[patch_idx][2];
                            weights[local] = pd.weights[patch_idx];
                            dofs[local] = (dof_offset + patch_idx) as u32;
                        }
                    }
                }

                elems.push(fem_linalg_gpu::GpuIgaBezier3DElement {
                    cpts,
                    weights,
                    dofs,
                    _pad: 0,
                });
            }
        }
    }

    elems
}

// ─── Validate supported degree ranges ────────────────────────────────────────

fn check_degree_2d(p: usize, q: usize) {
    assert!(p <= 3 && q <= 3,
        "GPU IGA 2D only supports degrees p,q ≤ 3, got p={p}, q={q}");
}

fn check_degree_3d(p: usize, q: usize, r: usize) {
    assert!(p <= 2 && q <= 2 && r <= 2,
        "GPU IGA 3D only supports degrees p,q,r ≤ 2, got p={p}, q={q}, r={r}");
}

// ─── 2-D diffusion ───────────────────────────────────────────────────────────

/// Assemble 2-D IGA diffusion stiffness matrix on GPU.
///
/// Requires the `gpu` feature and `wgpu` context with `SHADER_F64`.
/// Supports uniform knot vectors with degree `p,q ≤ 3`.
#[cfg(feature = "gpu")]
pub fn assemble_iga_diffusion_2d_bezier_gpu(
    gpu: &fem_linalg_gpu::context::GpuContext,
    mesh: &NurbsMesh2D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_2d(pd)
            .expect("compute_extraction_2d failed");
        check_degree_2d(ext.degree_u, ext.degree_v);

        let gpu_elems = build_gpu_elems_2d(pd, &ext, dof_offset);

        let triplets = fem_linalg_gpu::assemble_iga_bezier_diffusion_2d(
            gpu,
            &gpu_elems,
            ext.degree_u as u32,
            ext.degree_v as u32,
            quad_order as u32,
        );

        for (r, c, v) in triplets {
            if v != 0.0 {
                coo.add(r as usize, c as usize, kappa * v);
            }
        }

        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

// ─── 2-D mass ────────────────────────────────────────────────────────────────

/// Assemble 2-D IGA mass matrix on GPU.
#[cfg(feature = "gpu")]
pub fn assemble_iga_mass_2d_bezier_gpu(
    gpu: &fem_linalg_gpu::context::GpuContext,
    mesh: &NurbsMesh2D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_2d(pd)
            .expect("compute_extraction_2d failed");
        check_degree_2d(ext.degree_u, ext.degree_v);

        let gpu_elems = build_gpu_elems_2d(pd, &ext, dof_offset);

        let triplets = fem_linalg_gpu::assemble_iga_bezier_mass_2d(
            gpu,
            &gpu_elems,
            ext.degree_u as u32,
            ext.degree_v as u32,
            quad_order as u32,
        );

        for (r, c, v) in triplets {
            if v != 0.0 {
                coo.add(r as usize, c as usize, rho * v);
            }
        }

        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

// ─── 3-D diffusion ───────────────────────────────────────────────────────────

/// Assemble 3-D IGA diffusion stiffness matrix on GPU.
///
/// Supports uniform knot vectors with degree `p,q,r ≤ 2`.
#[cfg(feature = "gpu")]
pub fn assemble_iga_diffusion_3d_bezier_gpu(
    gpu: &fem_linalg_gpu::context::GpuContext,
    mesh: &NurbsMesh3D,
    kappa: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_3d(pd)
            .expect("compute_extraction_3d failed");
        check_degree_3d(ext.degree_u, ext.degree_v, ext.degree_w);

        let gpu_elems = build_gpu_elems_3d(pd, &ext, dof_offset);

        let triplets = fem_linalg_gpu::assemble_iga_bezier_diffusion_3d(
            gpu,
            &gpu_elems,
            ext.degree_u as u32,
            ext.degree_v as u32,
            ext.degree_w as u32,
            quad_order as u32,
        );

        for (r, c, v) in triplets {
            if v != 0.0 {
                coo.add(r as usize, c as usize, kappa * v);
            }
        }

        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

// ─── 3-D mass ────────────────────────────────────────────────────────────────

/// Assemble 3-D IGA mass matrix on GPU.
#[cfg(feature = "gpu")]
pub fn assemble_iga_mass_3d_bezier_gpu(
    gpu: &fem_linalg_gpu::context::GpuContext,
    mesh: &NurbsMesh3D,
    rho: f64,
    quad_order: u8,
) -> CsrMatrix<f64> {
    let n_total: usize = mesh.patches.iter().map(|p| p.control_pts.len()).sum();
    let mut coo = CooMatrix::<f64>::new(n_total, n_total);

    let mut dof_offset = 0usize;
    for pd in &mesh.patches {
        let ext = bezier_extraction::compute_extraction_3d(pd)
            .expect("compute_extraction_3d failed");
        check_degree_3d(ext.degree_u, ext.degree_v, ext.degree_w);

        let gpu_elems = build_gpu_elems_3d(pd, &ext, dof_offset);

        let triplets = fem_linalg_gpu::assemble_iga_bezier_mass_3d(
            gpu,
            &gpu_elems,
            ext.degree_u as u32,
            ext.degree_v as u32,
            ext.degree_w as u32,
            quad_order as u32,
        );

        for (r, c, v) in triplets {
            if v != 0.0 {
                coo.add(r as usize, c as usize, rho * v);
            }
        }

        dof_offset += pd.control_pts.len();
    }

    coo.into_csr()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_element::iga::{NurbsKnotVector, NurbsMesh2D, NurbsMesh3D};

    /// Build a uniform degree-2 patch on [0,1]² with 3×3 elements
    fn make_test_patch_2d(n_elems_u: usize, n_elems_v: usize) -> NurbsMesh2D {
        let p = 2;
        let kv_u = NurbsKnotVector::uniform(p, n_elems_u);
        let kv_v = NurbsKnotVector::uniform(p, n_elems_v);
        let nu = kv_u.n_basis();
        let nv = kv_v.n_basis();
        let n_dof = nu * nv;
        let ctrl: Vec<[f64; 2]> = (0..n_dof)
            .map(|idx| {
                let i = idx % nu;
                let j = idx / nu;
                [i as f64 / (nu - 1) as f64, j as f64 / (nv - 1) as f64]
            })
            .collect();
        NurbsMesh2D::single_patch(kv_u, kv_v, ctrl, vec![1.0; n_dof])
    }

    /// Build a uniform degree-1 patch on [0,1]³ with 2×2×2 elements
    fn make_test_patch_3d() -> NurbsMesh3D {
        let p = 1;
        let n = 4;
        let kv = NurbsKnotVector::uniform(p, n - p);
        let n_ctrl = n * n * n;
        let ctrl: Vec<[f64; 3]> = (0..n_ctrl)
            .map(|idx| {
                let i = idx % n;
                let j = (idx / n) % n;
                let k = idx / (n * n);
                [i as f64 / (n - 1) as f64,
                 j as f64 / (n - 1) as f64,
                 k as f64 / (n - 1) as f64]
            })
            .collect();
        NurbsMesh3D::single_patch(kv.clone(), kv.clone(), kv.clone(), ctrl, vec![1.0; n_ctrl])
    }

    /// CPU reference: nonempty_spans helper
    #[test]
    fn nonempty_spans_works() {
        let kv = NurbsKnotVector::uniform(2, 3);
        let spans = nonempty_spans(&kv.knots);
        assert_eq!(spans.len(), 3);
    }

    /// Check that 2D GPU element building matches DOF layout
    #[cfg(feature = "gpu")]
    #[test]
    fn build_gpu_elems_2d_basic() {
        let make_single_patch = || {
            let p = 2;
            let n_elems = 3;
            let kv_u = NurbsKnotVector::uniform(p, n_elems);
            let kv_v = NurbsKnotVector::uniform(p, n_elems);
            let nu = kv_u.n_basis();
            let nv = kv_v.n_basis();
            let n_dof = nu * nv;
            let ctrl = (0..n_dof).map(|idx| {
                let i = idx % nu;
                let j = idx / nu;
                [i as f64 / (nu - 1) as f64, j as f64 / (nv - 1) as f64]
            }).collect();
            let pd = fem_element::iga::NurbsPatch2DData {
                kv_u, kv_v,
                control_pts: ctrl,
                weights: vec![1.0; n_dof],
                tag: 1,
            };
            let ext = bezier_extraction::compute_extraction_2d(&pd).unwrap();
            (pd, ext)
        };
        let (pd, ext) = make_single_patch();
        let elems = build_gpu_elems_2d(&pd, &ext, 0);
        // 3×3 = 9 elements
        assert_eq!(elems.len(), 9);
        // n_local = (2+1)*(2+1) = 9
        for e in &elems {
            // All weights should be 1.0
            for w in e.weights.iter().take(9) {
                assert!((w - 1.0).abs() < 1e-14);
            }
        }
    }

    /// Compare GPU assembly output with CPU Bezier assembly for a small 2D
    /// uniform patch.  Skips gracefully when no GPU or no SHADER_F64.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_vs_cpu_2d_diffusion() {
        use crate::iga::iga_bezier::assemble_iga_diffusion_2d_bezier;

        let mesh = make_test_patch_2d(3, 3);
        let k_cpu = assemble_iga_diffusion_2d_bezier(&mesh, 1.0, 4);

        let gpu = match fem_linalg_gpu::context::GpuContext::new_sync() {
            Ok(g) => g,
            Err(_) => { eprintln!("no GPU adapter, skipping"); return; }
        };
        if !gpu.features.native_f64 { eprintln!("no f64, skipping"); return; }

        let k_gpu = assemble_iga_diffusion_2d_bezier_gpu(&gpu, &mesh, 1.0, 4);

        assert_eq!(k_gpu.nrows, k_cpu.nrows);
        let n = k_gpu.nrows;
        for i in 0..n {
            for ptr in k_gpu.row_ptr[i]..k_gpu.row_ptr[i + 1] {
                let j = k_gpu.col_idx[ptr] as usize;
                let diff = (k_gpu.values[ptr] - k_cpu.get(i, j)).abs();
                if diff > 1e-10 {
                    panic!("K[{i},{j}]: gpu={:.14e} cpu={:.14e} diff={:.2e}",
                        k_gpu.values[ptr], k_cpu.get(i, j), diff);
                }
            }
        }
    }

    /// Structural test: 2D mass matrix
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_vs_cpu_2d_mass() {
        use crate::iga::iga_bezier::assemble_iga_mass_2d_bezier;

        let mesh = make_test_patch_2d(3, 3);
        let m_cpu = assemble_iga_mass_2d_bezier(&mesh, 1.0, 4);

        let gpu = match fem_linalg_gpu::context::GpuContext::new_sync() {
            Ok(g) => g,
            Err(_) => { eprintln!("no GPU, skipping"); return; }
        };
        if !gpu.features.native_f64 { eprintln!("no f64, skipping"); return; }

        let m_gpu = assemble_iga_mass_2d_bezier_gpu(&gpu, &mesh, 1.0, 4);

        assert_eq!(m_gpu.nrows, m_cpu.nrows);
        let n = m_gpu.nrows;
        for i in 0..n {
            for ptr in m_gpu.row_ptr[i]..m_gpu.row_ptr[i + 1] {
                let j = m_gpu.col_idx[ptr] as usize;
                let diff = (m_gpu.values[ptr] - m_cpu.get(i, j)).abs();
                if diff > 1e-10 {
                    panic!("M[{i},{j}]: diff={:.2e}", diff);
                }
            }
        }
    }

    /// Structural test: 3D diffusion
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_vs_cpu_3d_diffusion() {
        use crate::iga::iga_bezier::assemble_iga_diffusion_3d_bezier;

        let mesh = make_test_patch_3d();
        let k_cpu = assemble_iga_diffusion_3d_bezier(&mesh, 1.0, 3);

        let gpu = match fem_linalg_gpu::context::GpuContext::new_sync() {
            Ok(g) => g,
            Err(_) => { eprintln!("no GPU, skipping"); return; }
        };
        if !gpu.features.native_f64 { eprintln!("no f64, skipping"); return; }

        let k_gpu = assemble_iga_diffusion_3d_bezier_gpu(&gpu, &mesh, 1.0, 3);

        assert_eq!(k_gpu.nrows, k_cpu.nrows);
        let n = k_gpu.nrows;
        for i in 0..n {
            for ptr in k_gpu.row_ptr[i]..k_gpu.row_ptr[i + 1] {
                let j = k_gpu.col_idx[ptr] as usize;
                let diff = (k_gpu.values[ptr] - k_cpu.get(i, j)).abs();
                if diff > 1e-10 {
                    panic!("K3D[{i},{j}]: gpu={:.14e} cpu={:.14e} diff={:.2e}",
                        k_gpu.values[ptr], k_cpu.get(i, j), diff);
                }
            }
        }
    }
}
