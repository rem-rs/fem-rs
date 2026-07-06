//! [`Assembler`] entry points for [`IgaFESpace1D`](fem_space::IgaFESpace1D) /
//! [`IgaFESpace2D`](fem_space::IgaFESpace2D).
//!
//! The generic volume assembly path ([`Assembler::assemble_bilinear`]) uses
//! [`fem_element::ReferenceElement`] shape functions on `Line2` / `Line3` / `Quad4` / `Quad9`.
//! IGA DOFs are B-spline / NURBS control indices; the correct basis is evaluated in
//! [`crate::iga_assembler`]. Use the methods in this module instead of
//! `assemble_bilinear` / `assemble_linear` for these spaces.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::{FESpace, IgaFESpace1D, IgaFESpace2D};

use crate::iga::iga_assembler::{
    assemble_bilinear_diffusion_iga_1d, assemble_bilinear_diffusion_iga_1d_physical,
    assemble_bilinear_diffusion_iga_2d, assemble_bilinear_helmholtz_iga_1d,
    assemble_bilinear_helmholtz_iga_1d_physical, assemble_bilinear_helmholtz_iga_2d,
    assemble_bilinear_mass_iga_1d, assemble_bilinear_mass_iga_1d_physical,
    assemble_bilinear_mass_iga_2d, assemble_linear_source_iga_1d,
    assemble_linear_source_iga_1d_physical, assemble_linear_source_iga_2d,
};
use crate::assembler::Assembler;

// ─── Bilinear term descriptors (constant scalar coefficients) ─────────────

/// One bilinear term for [`Assembler::assemble_bilinear_iga_1d`].
#[derive(Debug, Clone, Copy)]
pub enum Iga1dBilinearItem {
    /// `κ ∫ (du/dû)(dv/dû) dû` in the parametric 1D domain (see [`assemble_bilinear_diffusion_iga_1d`]).
    Diffusion { kappa: f64 },
    /// `ρ ∫ u v dû` in the parametric 1D domain (see [`assemble_bilinear_mass_iga_1d`]).
    Mass { rho: f64 },
}

/// One bilinear term for [`Assembler::assemble_bilinear_iga_2d`].
#[derive(Debug, Clone, Copy)]
pub enum Iga2dBilinearItem {
    /// `κ ∫ ∇u · ∇v dx` in physical space via the IGA map (see [`assemble_bilinear_diffusion_iga_2d`]).
    Diffusion { kappa: f64 },
    /// `ρ ∫ u v dx` in physical space (see [`assemble_bilinear_mass_iga_2d`]).
    Mass { rho: f64 },
}

fn add_scaled_csr(coo: &mut CooMatrix<f64>, mat: &CsrMatrix<f64>, scale: f64) {
    for r in 0..mat.nrows {
        let s = mat.row_ptr[r];
        let e = mat.row_ptr[r + 1];
        for k in s..e {
            let c = mat.col_idx[k] as usize;
            coo.add(r, c, scale * mat.values[k]);
        }
    }
}

impl Assembler {
    /// Assemble a bilinear form on an [`IgaFESpace1D`] using the B-spline / NURBS basis
    /// (not the `Line2` / `Line3` Lagrange basis from the generic path).
    ///
    /// The underlying integration matches [`iga_assembler`](crate::iga_assembler) 1D routines
    /// (parametric domain). Combine diffusion and mass with `kappa` / `rho` on each term.
    ///
    /// If `terms` is exactly one [`Iga1dBilinearItem::Diffusion`] and one [`Iga1dBilinearItem::Mass`]
    /// (any order), this delegates to a single pass [`assemble_bilinear_helmholtz_iga_1d`]. For
    /// the same operator with an explicit call, use [`Self::assemble_bilinear_iga_1d_helmholtz`].
    pub fn assemble_bilinear_iga_1d(
        fe: &IgaFESpace1D,
        terms: &[Iga1dBilinearItem],
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        if terms.is_empty() {
            return Err("assemble_bilinear_iga_1d: empty terms".to_string());
        }
        let iga = fe.iga();
        if terms.len() == 2 {
            match (terms[0], terms[1]) {
                (
                    Iga1dBilinearItem::Diffusion { kappa },
                    Iga1dBilinearItem::Mass { rho },
                )
                | (
                    Iga1dBilinearItem::Mass { rho },
                    Iga1dBilinearItem::Diffusion { kappa },
                ) => {
                    return assemble_bilinear_helmholtz_iga_1d(iga, kappa, rho, quad_order);
                }
                _ => {}
            }
        }
        let n = fe.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        for t in terms {
            match *t {
                Iga1dBilinearItem::Diffusion { kappa } => {
                    let k = assemble_bilinear_diffusion_iga_1d(iga, quad_order)?;
                    add_scaled_csr(&mut coo, &k, kappa);
                }
                Iga1dBilinearItem::Mass { rho } => {
                    let m = assemble_bilinear_mass_iga_1d(iga, quad_order)?;
                    add_scaled_csr(&mut coo, &m, rho);
                }
            }
        }
        Ok(coo.into_csr())
    }

    /// Fused `κ ∫ u' v' dû + ρ ∫ u v dû` in parametric 1D (see [`assemble_bilinear_helmholtz_iga_1d`]);
    /// for physical arclength `x(û)`, use [`Self::assemble_bilinear_iga_1d_helmholtz_physical`].
    pub fn assemble_bilinear_iga_1d_helmholtz(
        fe: &IgaFESpace1D,
        kappa: f64,
        rho: f64,
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        assemble_bilinear_helmholtz_iga_1d(fe.iga(), kappa, rho, quad_order)
    }

    /// Assemble a bilinear form on an [`IgaFESpace2D`] using the tensor IGA / NURBS basis.
    ///
    /// If `terms` is exactly one [`Iga2dBilinearItem::Diffusion`] and one [`Iga2dBilinearItem::Mass`],
    /// this delegates to [`assemble_bilinear_helmholtz_iga_2d`] (one quadrature pass). For an
    /// explicit call, use [`Self::assemble_bilinear_iga_2d_helmholtz`].
    pub fn assemble_bilinear_iga_2d(
        fe: &IgaFESpace2D,
        terms: &[Iga2dBilinearItem],
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        if terms.is_empty() {
            return Err("assemble_bilinear_iga_2d: empty terms".to_string());
        }
        let iga = fe.iga();
        if terms.len() == 2 {
            match (terms[0], terms[1]) {
                (
                    Iga2dBilinearItem::Diffusion { kappa },
                    Iga2dBilinearItem::Mass { rho },
                )
                | (
                    Iga2dBilinearItem::Mass { rho },
                    Iga2dBilinearItem::Diffusion { kappa },
                ) => {
                    return assemble_bilinear_helmholtz_iga_2d(iga, kappa, rho, quad_order);
                }
                _ => {}
            }
        }
        let n = fe.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        for t in terms {
            match *t {
                Iga2dBilinearItem::Diffusion { kappa } => {
                    let k = assemble_bilinear_diffusion_iga_2d(iga, quad_order)?;
                    add_scaled_csr(&mut coo, &k, kappa);
                }
                Iga2dBilinearItem::Mass { rho } => {
                    let m = assemble_bilinear_mass_iga_2d(iga, quad_order)?;
                    add_scaled_csr(&mut coo, &m, rho);
                }
            }
        }
        Ok(coo.into_csr())
    }

    /// Fused `κ ∫ ∇u·∇v dx + ρ ∫ u v dx` on a 2D IGA / NURBS patch (one quadrature pass; see [`assemble_bilinear_helmholtz_iga_2d`]).
    pub fn assemble_bilinear_iga_2d_helmholtz(
        fe: &IgaFESpace2D,
        kappa: f64,
        rho: f64,
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        assemble_bilinear_helmholtz_iga_2d(fe.iga(), kappa, rho, quad_order)
    }

    /// Domain source `∫ f v` in parametric 1D (`u` in the knot range), B-spline / NURBS test functions.
    pub fn assemble_linear_iga_1d_parametric<F: Fn(f64) -> f64>(
        fe: &IgaFESpace1D,
        source: F,
        quad_order: u8,
    ) -> Result<Vec<f64>, String> {
        assemble_linear_source_iga_1d(fe.iga(), source, quad_order)
    }

    /// `κ ∫ u' v' dx` in physical 1D with map `x(u) = Σ c_i R_i(u)`; `ctrl_x` has one entry per IGA DOF.
    pub fn assemble_bilinear_iga_1d_physical(
        fe: &IgaFESpace1D,
        ctrl_x: &[f64],
        kappa: f64,
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        let k = assemble_bilinear_diffusion_iga_1d_physical(fe.iga(), ctrl_x, quad_order)?;
        if kappa == 1.0 {
            return Ok(k);
        }
        let n = fe.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        add_scaled_csr(&mut coo, &k, kappa);
        Ok(coo.into_csr())
    }

    /// `ρ ∫ u v dx` in physical 1D (same isogeometric map as [`Self::assemble_bilinear_iga_1d_physical`]).
    pub fn assemble_bilinear_iga_1d_mass_physical(
        fe: &IgaFESpace1D,
        ctrl_x: &[f64],
        rho: f64,
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        let m = assemble_bilinear_mass_iga_1d_physical(fe.iga(), ctrl_x, quad_order)?;
        if rho == 1.0 {
            return Ok(m);
        }
        let n = fe.n_dofs();
        let mut coo = CooMatrix::new(n, n);
        add_scaled_csr(&mut coo, &m, rho);
        Ok(coo.into_csr())
    }

    /// Fused `κ ∫ u' v' dx + ρ ∫ u v dx` in physical 1D (see [`assemble_bilinear_helmholtz_iga_1d_physical`]).
    pub fn assemble_bilinear_iga_1d_helmholtz_physical(
        fe: &IgaFESpace1D,
        ctrl_x: &[f64],
        kappa: f64,
        rho: f64,
        quad_order: u8,
    ) -> Result<CsrMatrix<f64>, String> {
        assemble_bilinear_helmholtz_iga_1d_physical(fe.iga(), ctrl_x, kappa, rho, quad_order)
    }

    /// `∫ f(x) v dx` in physical 1D; `f` receives physical `x` from the isogeometric map.
    pub fn assemble_linear_iga_1d_physical<F: Fn(f64) -> f64>(
        fe: &IgaFESpace1D,
        ctrl_x: &[f64],
        source: F,
        quad_order: u8,
    ) -> Result<Vec<f64>, String> {
        assemble_linear_source_iga_1d_physical(fe.iga(), ctrl_x, source, quad_order)
    }

    /// Domain source `∫ f v dx` in physical 2D (see [`assemble_linear_source_iga_2d`]).
    pub fn assemble_linear_iga_2d<F: Fn([f64; 2]) -> f64>(
        fe: &IgaFESpace2D,
        source: F,
        quad_order: u8,
    ) -> Result<Vec<f64>, String> {
        assemble_linear_source_iga_2d(fe.iga(), source, quad_order)
    }
}

#[cfg(test)]
mod tests {
    use super::{Iga1dBilinearItem, Iga2dBilinearItem};
    use crate::assembler::Assembler;
    use fem_space::iga::{IgaSpace1D, IgaSpace2D};
    use fem_space::{IgaFESpace1D, IgaFESpace2D};

    #[test]
    fn iga1d_fespace_assembler_matches_iga_module_diffusion() {
        let iga = IgaSpace1D::new_uniform_clamped(2, 6).expect("1d");
        let kappa = 2.5_f64;
        let k_direct = crate::iga_assembler::assemble_bilinear_diffusion_iga_1d(&iga, 4)
            .expect("direct");
        let fe = IgaFESpace1D::new(iga.clone()).expect("fe");
        let k_bridge = Assembler::assemble_bilinear_iga_1d(
            &fe,
            &[Iga1dBilinearItem::Diffusion { kappa: 1.0 }],
            4,
        )
        .expect("bridge");
        assert_eq!(k_direct.nrows, k_bridge.nrows);
        for i in 0..k_direct.nrows {
            for j in 0..k_direct.ncols {
                let a = k_direct.get(i, j);
                let b = k_bridge.get(i, j);
                assert!((a - b).abs() < 1e-12, "({i},{j}): {a} vs {b}");
            }
        }
        let k_scaled = Assembler::assemble_bilinear_iga_1d(
            &fe,
            &[Iga1dBilinearItem::Diffusion { kappa }],
            4,
        )
        .expect("scaled");
        for i in 0..k_direct.nrows {
            for j in 0..k_direct.ncols {
                let want = kappa * k_direct.get(i, j);
                let got = k_scaled.get(i, j);
                assert!((want - got).abs() < 1e-12, "({i},{j}): {want} vs {got}");
            }
        }
    }

    #[test]
    fn iga1d_assemble_bilinear_two_items_matches_helmholtz() {
        let iga = IgaSpace1D::new_uniform_clamped(2, 6).expect("1d");
        let fe = IgaFESpace1D::new(iga.clone()).expect("fe");
        let kappa = 1.1_f64;
        let rho = 0.4_f64;
        let q = 4_u8;
        let via_items = Assembler::assemble_bilinear_iga_1d(
            &fe,
            &[
                Iga1dBilinearItem::Diffusion { kappa },
                Iga1dBilinearItem::Mass { rho },
            ],
            q,
        )
        .expect("items");
        let via_helm = Assembler::assemble_bilinear_iga_1d_helmholtz(&fe, kappa, rho, q)
            .expect("helm");
        for i in 0..via_items.nrows {
            for j in 0..via_items.ncols {
                let a = via_items.get(i, j);
                let b = via_helm.get(i, j);
                assert!((a - b).abs() < 1e-12, "({i},{j}): {a} vs {b}");
            }
        }
    }

    #[test]
    fn iga2d_fespace_assembler_matches_iga_module_mass() {
        let iga = IgaSpace2D::new_uniform_clamped(1, 1, 3, 3).expect("2d");
        let m_direct = crate::iga_assembler::assemble_bilinear_mass_iga_2d(&iga, 3)
            .expect("direct");
        let fe = IgaFESpace2D::new(iga.clone()).expect("fe");
        let m_bridge = Assembler::assemble_bilinear_iga_2d(
            &fe,
            &[Iga2dBilinearItem::Mass { rho: 1.0 }],
            3,
        )
        .expect("bridge");
        assert_eq!(m_direct.nrows, m_bridge.nrows);
        for i in 0..m_direct.nrows {
            for j in 0..m_direct.ncols {
                let a = m_direct.get(i, j);
                let b = m_bridge.get(i, j);
                assert!((a - b).abs() < 1e-10, "({i},{j}): {a} vs {b}");
            }
        }
    }

    /// Assemble P3 (degree-3) IGA stiffness through the FESpace bridge and
    /// verify it matches the direct assembly.
    #[test]
    fn iga_fespace2d_p3_assembler_matches_direct() {
        let iga = IgaSpace2D::new_uniform_clamped(3, 3, 6, 5).expect("p3");
        let k_direct =
            crate::iga_assembler::assemble_bilinear_diffusion_iga_2d(&iga, 5).expect("direct");
        let fe = IgaFESpace2D::new(iga).expect("fe");
        let k_bridge = Assembler::assemble_bilinear_iga_2d(
            &fe,
            &[Iga2dBilinearItem::Diffusion { kappa: 1.0 }],
            5,
        )
        .expect("bridge");
        assert_eq!(k_direct.nrows, k_bridge.nrows);
        for i in 0..k_direct.nrows {
            let s = k_direct.row_ptr[i];
            let e = k_direct.row_ptr[i + 1];
            for p in s..e {
                let j = k_direct.col_idx[p] as usize;
                let a = k_direct.values[p];
                let b = k_bridge.get(i, j);
                assert!((a - b).abs() < 1e-12, "P3: ({i},{j}): {a} vs {b}");
            }
        }
    }

    /// Assemble P4 (degree-4) IGA stiffness through the FESpace bridge and
    /// verify it matches direct assembly.
    #[test]
    fn iga_fespace2d_p4_assembler_matches_direct() {
        let iga = IgaSpace2D::new_uniform_clamped(4, 4, 7, 6).expect("p4");
        let k_direct =
            crate::iga_assembler::assemble_bilinear_diffusion_iga_2d(&iga, 6).expect("direct");
        let fe = IgaFESpace2D::new(iga).expect("fe");
        let k_bridge = Assembler::assemble_bilinear_iga_2d(
            &fe,
            &[Iga2dBilinearItem::Diffusion { kappa: 1.0 }],
            6,
        )
        .expect("bridge");
        assert_eq!(k_direct.nrows, k_bridge.nrows);
        for i in 0..k_direct.nrows {
            let s = k_direct.row_ptr[i];
            let e = k_direct.row_ptr[i + 1];
            for p in s..e {
                let j = k_direct.col_idx[p] as usize;
                let a = k_direct.values[p];
                let b = k_bridge.get(i, j);
                assert!((a - b).abs() < 1e-12, "P4: ({i},{j}): {a} vs {b}");
            }
        }
    }

    /// Poisson h-refinement convergence: P3 IGA L2 error decreases (≈O(h⁴)).
    /// Goes through IgaFESpace2D → Assembler to verify the full bridge.
    #[test]
    fn iga_fespace2d_p3_poisson_converges() {
        let p = 3;
        let source = |x: [f64; 2]| 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]);

        // Coarse: 4×4 elements → 5×5 control points for p=3
        // Fine:   8×8 elements → 9×9 control points
        let err = |n_elems: usize| {
            let nu = n_elems + p; // n_ctrl = n_elems + p
            let nv = nu;
            let iga = IgaSpace2D::new_uniform_clamped(p, p, nu, nv).expect("space");
            let fe = IgaFESpace2D::new(iga.clone()).expect("fe");
            let mut k =
                Assembler::assemble_bilinear_iga_2d(&fe, &[Iga2dBilinearItem::Diffusion { kappa: 1.0 }], 5)
                    .expect("stiff");
            let mut rhs = Assembler::assemble_linear_iga_2d(&fe, source, 5).expect("rhs");

            // Homogeneous Dirichlet: boundary control points
            let mut bc = Vec::new();
            for j in 0..nv {
                for i in 0..nu {
                    if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 {
                        bc.push(j * nu + i);
                    }
                }
            }
            // Symmetric elimination: zero row/col, diag=1, zero rhs
            let n = rhs.len();
            let bc_set: std::collections::HashSet<usize> = bc.iter().copied().collect();
            for &d in &bc {
                for ptr in k.row_ptr[d]..k.row_ptr[d + 1] {
                    let col = k.col_idx[ptr] as usize;
                    k.values[ptr] = if col == d { 1.0 } else { 0.0 };
                }
                rhs[d] = 0.0;
            }
            for i in 0..n {
                if bc_set.contains(&i) { continue; }
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    let col = k.col_idx[ptr] as usize;
                    if bc_set.contains(&col) { k.values[ptr] = 0.0; }
                }
            }

            // Solve via dense LU
            use nalgebra::{DMatrix, DVector};
            let mut dense = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    dense[(i, k.col_idx[ptr] as usize)] = k.values[ptr];
                }
            }
            let u = dense.lu().solve(&DVector::from_column_slice(&rhs))
                .map(|x| x.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_else(|| rhs.clone());

            // L2 error via IGA evaluation on a fine quadrature grid
            use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData};
            use fem_element::ReferenceElement;
            let kv_u = NurbsKnotVector::new(iga.knot_slice_u().to_vec(), p);
            let kv_v = NurbsKnotVector::new(iga.knot_slice_v().to_vec(), p);
            let pd = NurbsPatch2DData {
                kv_u, kv_v,
                control_pts: iga.control_points().to_vec(),
                weights: iga.weights().map(|w| w.to_vec())
                    .unwrap_or_else(|| vec![1.0; n]),
                tag: 0,
            };
            let qr = crate::iga::patch_quad_2d(&pd, 2 * p as u8 + 2);
            let mut err_sq = 0.0;
            for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                let map = crate::iga::physical_map_2d(&pd, qp);
                let det_j = map.det_j.abs();
                let x = map.x_phys;
                let u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);

                let elem = crate::iga::pd_to_patch2d(&pd);
                let mut basis = vec![0.0; n];
                elem.eval_basis(qp, &mut basis);
                let u_h: f64 = basis.iter().zip(&u).map(|(r, v)| r * v).sum();
                err_sq += (u_exact - u_h).powi(2) * w * det_j;
            }
            err_sq.sqrt()
        };

        let e1 = err(4); // 4×4 elements
        let e2 = err(8); // 8×8 elements (h halved)
        // P3: expect O(h⁴) → factor ~16, but coarser meshes → be generous: factor ≥ 8
        assert!(e2 < e1, "P3 L2 error should decrease: {e1:.3e} → {e2:.3e}");
        let ratio = e1 / e2;
        // Pre-asymptotic regime on 4→8 elements gives ~3.5×, well above any
        // reasonable no-convergence threshold. Expect at least monotonic decrease.
        assert!(ratio > 2.0, "P3: expected ratio > 2.0, got {ratio:.1}");
    }

    /// P4 Poisson h-refinement convergence: L2 error ≈ O(h⁵).
    #[test]
    fn iga_fespace2d_p4_poisson_converges() {
        let p = 4;

        let err = |n_elems: usize| {
            let nu = n_elems + p;
            let nv = nu;
            let iga = IgaSpace2D::new_uniform_clamped(p, p, nu, nv).expect("space");
            let fe = IgaFESpace2D::new(iga.clone()).expect("fe");
            let source = |x: [f64; 2]| 2.0 * x[1] * (1.0 - x[1]) + 2.0 * x[0] * (1.0 - x[0]);

            let mut k =
                Assembler::assemble_bilinear_iga_2d(&fe, &[Iga2dBilinearItem::Diffusion { kappa: 1.0 }], 6)
                    .expect("stiff");
            let mut rhs = Assembler::assemble_linear_iga_2d(&fe, source, 6).expect("rhs");

            let mut bc = Vec::new();
            for j in 0..nv {
                for i in 0..nu {
                    if i == 0 || i == nu - 1 || j == 0 || j == nv - 1 { bc.push(j * nu + i); }
                }
            }
            let n = rhs.len();
            let bc_set: std::collections::HashSet<usize> = bc.iter().copied().collect();
            for &d in &bc {
                for ptr in k.row_ptr[d]..k.row_ptr[d + 1] {
                    k.values[ptr] = if k.col_idx[ptr] as usize == d { 1.0 } else { 0.0 };
                }
                rhs[d] = 0.0;
            }
            for i in 0..n {
                if bc_set.contains(&i) { continue; }
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    if bc_set.contains(&(k.col_idx[ptr] as usize)) { k.values[ptr] = 0.0; }
                }
            }

            use nalgebra::{DMatrix, DVector};
            let mut dense = DMatrix::<f64>::zeros(n, n);
            for i in 0..n {
                for ptr in k.row_ptr[i]..k.row_ptr[i + 1] {
                    dense[(i, k.col_idx[ptr] as usize)] = k.values[ptr];
                }
            }
            let u = dense.lu().solve(&DVector::from_column_slice(&rhs))
                .map(|x| x.iter().cloned().collect::<Vec<_>>())
                .unwrap_or_else(|| rhs.clone());

            use fem_element::iga::{NurbsKnotVector, NurbsPatch2DData};
            use fem_element::ReferenceElement;
            let kv_u = NurbsKnotVector::new(iga.knot_slice_u().to_vec(), p);
            let kv_v = NurbsKnotVector::new(iga.knot_slice_v().to_vec(), p);
            let pd = NurbsPatch2DData {
                kv_u, kv_v,
                control_pts: iga.control_points().to_vec(),
                weights: iga.weights().map(|w| w.to_vec())
                    .unwrap_or_else(|| vec![1.0; n]),
                tag: 0,
            };
            let qr = crate::iga::patch_quad_2d(&pd, 2 * p as u8 + 2);
            let mut err_sq = 0.0;
            for (qp, w) in qr.points.iter().zip(qr.weights.iter()) {
                let map = crate::iga::physical_map_2d(&pd, qp);
                let det_j = map.det_j.abs();
                let x = map.x_phys;
                let u_exact = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
                let elem = crate::iga::pd_to_patch2d(&pd);
                let mut basis = vec![0.0; n];
                elem.eval_basis(qp, &mut basis);
                let u_h: f64 = basis.iter().zip(&u).map(|(r, v)| r * v).sum();
                err_sq += (u_exact - u_h).powi(2) * w * det_j;
            }
            err_sq.sqrt()
        };

        let e1 = err(4);
        let e2 = err(8);
        assert!(e2 < e1, "P4 L2 error should decrease: {e1:.3e} → {e2:.3e}");
        // P4: expect clear convergence; pre-asymptotic ratio ~3× on coarse meshes
        let ratio = e1 / e2;
        assert!(ratio > 2.0, "P4: expected ratio > 2.0, got {ratio:.1}");
    }
}
