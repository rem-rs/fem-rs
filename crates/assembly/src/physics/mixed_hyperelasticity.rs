//! Mixed u/p incompressible hyperelasticity (MFEM ex19).
//!
//! Implements the neo-Hookean incompressible hyperelastic model using a
//! mixed displacement/pressure (u/p) formulation:
//!
//! ```text
//! R_u(i,a) = ∫ (μ·F_{iJ} - p·F^{-T}_{iJ}) · ∂φ_a/∂X_J  dx
//! R_p(m)   = ∫ (J - 1) · ψ_m                              dx
//! ```
//!
//! with block Jacobian
//! ```text
//! J = [K_uu  K_up]
//!     [K_pu   0  ]
//! ```
//!
//! The formulation uses Taylor-Hood elements: VectorH1^dim for displacement
//! (order k) and H1 for pressure (order k-1).
//!
//! # Reference
//! MFEM ex19 — Quasi-static incompressible neo-Hookean hyperelasticity

#![allow(non_snake_case)]

use nalgebra::DMatrix;

use fem_linalg::{BlockMatrix, CooMatrix, CsrMatrix};
use fem_mesh::{geometry_jacobian, xform_grads, topology::MeshTopology};

/// MFEM `DenseMatrix::Det` closed form (densemat.cpp, row-0 cofactor chain).
/// ulp-path parity with C++ is a hard requirement of the ex19 BIT run —
/// `nalgebra`'s determinant follows a different elimination order and leaves
/// 6th-digit dust in the printed GMRES tails.
fn mfem_det(F: &DMatrix<f64>) -> f64 {
    match F.nrows() {
        1 => F[(0, 0)],
        2 => F[(0, 0)] * F[(1, 1)] - F[(0, 1)] * F[(1, 0)],
        3 => {
            F[(0, 0)] * (F[(1, 1)] * F[(2, 2)] - F[(1, 2)] * F[(2, 1)])
                - F[(0, 1)] * (F[(1, 0)] * F[(2, 2)] - F[(1, 2)] * F[(2, 0)])
                + F[(0, 2)] * (F[(1, 0)] * F[(2, 1)] - F[(1, 1)] * F[(2, 0)])
        }
        // The form is declared for dim ≤ 3 (matching MFEM's own
        // CalcInverseTranspose domain); larger dims never reach here.
        _ => unreachable!("MixedHyperelasticityForm supports dim 1..=3"),
    }
}

/// MFEM `CalcInverseTranspose` (linalg/densemat.cpp:2721) verbatim —
/// one `1/det` division, then per-cofactor products in the C++ order.
fn mfem_inverse_transpose(F: &DMatrix<f64>) -> DMatrix<f64> {
    let dim = F.nrows();
    let mut inva = DMatrix::<f64>::zeros(dim, dim);
    let t = 1.0 / mfem_det(F);
    match dim {
        1 => inva[(0, 0)] = 1.0 / F[(0, 0)],
        2 => {
            inva[(0, 0)] = F[(1, 1)] * t;
            inva[(1, 0)] = -F[(0, 1)] * t;
            inva[(0, 1)] = -F[(1, 0)] * t;
            inva[(1, 1)] = F[(0, 0)] * t;
        }
        3 => {
            inva[(0, 0)] = (F[(1, 1)] * F[(2, 2)] - F[(1, 2)] * F[(2, 1)]) * t;
            inva[(1, 0)] = (F[(0, 2)] * F[(2, 1)] - F[(0, 1)] * F[(2, 2)]) * t;
            inva[(2, 0)] = (F[(0, 1)] * F[(1, 2)] - F[(0, 2)] * F[(1, 1)]) * t;

            inva[(0, 1)] = (F[(1, 2)] * F[(2, 0)] - F[(1, 0)] * F[(2, 2)]) * t;
            inva[(1, 1)] = (F[(0, 0)] * F[(2, 2)] - F[(0, 2)] * F[(2, 0)]) * t;
            inva[(2, 1)] = (F[(0, 2)] * F[(1, 0)] - F[(0, 0)] * F[(1, 2)]) * t;

            inva[(0, 2)] = (F[(1, 0)] * F[(2, 1)] - F[(1, 1)] * F[(2, 0)]) * t;
            inva[(1, 2)] = (F[(0, 1)] * F[(2, 0)] - F[(0, 0)] * F[(2, 1)]) * t;
            inva[(2, 2)] = (F[(0, 0)] * F[(1, 1)] - F[(0, 1)] * F[(1, 0)]) * t;
        }
        _ => unreachable!("MixedHyperelasticityForm supports dim 1..=3"),
    }
    inva
}

/// Mixed u/p incompressible neo-Hookean form (MFEM: RubberOperator).
///
/// Pre-computes element DOF tables on construction.
/// Use [`MixedHyperelasticityForm::residual`] and [`::jacobian_blocks`]
/// to assemble the nonlinear system.
pub struct MixedHyperelasticityForm {
    mesh: Box<dyn MeshTopology + Send + Sync>,
    dim: usize,
    order: u8,
    p_order: u8,
    quad_order: u8,
    mu: f64,
    nu: usize,
    np: usize,
    ns: usize,
    elem_dofs_u: Vec<Vec<usize>>,
    elem_dofs_p: Vec<Vec<usize>>,
    dirichlet: Vec<(usize, f64)>,
}

impl MixedHyperelasticityForm {
    /// Build the mixed form from FE spaces and material parameters.
    ///
    /// `dim` — spatial dimension (2 or 3).
    /// `order` — displacement FE order (pressure uses order-1, Taylor-Hood).
    /// `mu` — shear modulus.
    /// `dirichlet` — list of (global_dof, value) for essential BCs.
    pub fn new(
        mesh: Box<dyn MeshTopology + Send + Sync>,
        dim: usize,
        order: u8,
        p_order: u8,
        mu: f64,
        nu: usize,
        np: usize,
        ns: usize,
        elem_dofs_u: Vec<Vec<usize>>,
        elem_dofs_p: Vec<Vec<usize>>,
        dirichlet: Vec<(usize, f64)>,
    ) -> Self {
        let quad_order = 2 * order + 3;
        Self {
            mesh, dim, order, p_order, quad_order, mu,
            nu, np, ns,
            elem_dofs_u, elem_dofs_p, dirichlet,
        }
    }

    /// Number of displacement DOFs.
    pub fn nu(&self) -> usize { self.nu }

    /// Number of pressure DOFs.
    pub fn np(&self) -> usize { self.np }

    /// Total DOFs (u + p).
    pub fn n_dofs(&self) -> usize { self.nu + self.np }

    /// Compute the residual vector `[R_u; R_p]`.
    ///
    /// MFEM: `RubberOperator::Mult`
    pub fn residual(
        &self,
        u: &[f64],
        p: &[f64],
        ru: &mut [f64],
        rp: &mut [f64],
    ) {
        ru.fill(0.0);
        rp.fill(0.0);
        let ne = self.elem_dofs_u.len();
        for e in 0..ne {
            let et = self.mesh.element_type(e as u32);
            let ru_ref = et.ref_elem(self.order);
            let rp_ref = et.ref_elem(self.p_order);
            let n_du = ru_ref.n_dofs();
            let n_dp = rp_ref.n_dofs();
            let n_vd = n_du * self.dim;

            let eu = &self.elem_dofs_u[e];
            let ep = &self.elem_dofs_p[e];

            let mut ue = vec![0.0_f64; n_vd];
            for (k, &g) in eu.iter().enumerate() { ue[k] = u[g]; }
            let mut pe = vec![0.0_f64; n_dp];
            for (k, &g) in ep.iter().enumerate() { pe[k] = p[g]; }

            let q = ru_ref.quadrature(self.quad_order);
            let mut phi_u = vec![0.0_f64; n_du];
            let mut gr_u = vec![0.0_f64; n_du * self.dim];
            let mut gp_u = vec![0.0_f64; n_du * self.dim];
            let mut phi_p = vec![0.0_f64; n_dp];
            let mut fu_e = vec![0.0_f64; n_vd];
            let mut fp_e = vec![0.0_f64; n_dp];

            for (qi, xi) in q.points.iter().enumerate() {
                ru_ref.eval_basis(xi, &mut phi_u);
                ru_ref.eval_grad_basis(xi, &mut gr_u);
                rp_ref.eval_basis(xi, &mut phi_p);

                let (det_j, ji) = geometry_jacobian(&*self.mesh, e as u32, xi, self.dim);
                xform_grads(&ji, &gr_u, &mut gp_u, n_du, self.dim);
                // D696 batch 4: SIGNED — `ip.weight * Ttr.Weight()`
                // (nonlininteg.cpp:420/461/498 class); bitwise |det| on
                // valid meshes.
                let w = q.weights[qi] * det_j;

                // Deformation gradient F = I + ∇u
                let mut F = DMatrix::<f64>::identity(self.dim, self.dim);
                for k in 0..n_du {
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            F[(i, j)] += ue[k * self.dim + i] * gp_u[k * self.dim + j];
                        }
                    }
                }
                let dJ = mfem_det(&F);
                let FT = mfem_inverse_transpose(&F);

                let mut pres = 0.0;
                for k in 0..n_dp { pres += pe[k] * phi_p[k]; }

                // Momentum integrand, MFEM IncompressibleNeoHookeanIntegrator::
                // AssembleElementVector (nonlininteg.cpp:608-613):
                //   P = μ·dJ·F + (−p·dJ)·F⁻ᵀ;  P *= ip.weight*Tr.Weight();
                //   AddMultABt(DS_u, P, PMatO_u)
                // The `w` scaling happens **before** the mat-vec (the old form
                // applied it to the already-summed row), and AddMultABt adds one
                // `+=` per column `l` straight into the element vector — both
                // matter for the rounding sequence (rr157 lane2, source (3)).
                let mut P = DMatrix::<f64>::zeros(self.dim, self.dim);
                for i in 0..self.dim {
                    for j in 0..self.dim {
                        P[(i, j)] =
                            (self.mu * dJ * F[(i, j)] - pres * dJ * FT[(i, j)]) * w;
                    }
                }

                // R_u: MFEM `AddMultABt(DS_u, P, PMatO_u)` — accumulate
                // `DS_u(a,l)·P(i,l)` per column `l` (ascending), exactly as the
                // reference kernel does.
                for l in 0..self.dim {
                    for i in 0..self.dim {
                        let p_il = P[(i, l)];
                        for a in 0..n_du {
                            fu_e[a * self.dim + i] += gp_u[a * self.dim + l] * p_il;
                        }
                    }
                }

                // R_p: (J - 1) · ψ
                for m in 0..n_dp { fp_e[m] += w * (dJ - 1.0) * phi_p[m]; }
            }

            for (k, &g) in eu.iter().enumerate() { ru[g] += fu_e[k]; }
            for (k, &g) in ep.iter().enumerate() { rp[g] += fp_e[k]; }
        }

        // Zero residual at Dirichlet DOFs
        for &(dof, _) in &self.dirichlet { ru[dof] = 0.0; }
    }

    /// Assemble the block Jacobian `[K_uu, K_up; K_pu, 0]` with
    /// Dirichlet row-zeroing applied.
    ///
    /// Returns `(block_sizes, BlockMatrix)`.
    ///
    /// MFEM: `RubberOperator::GetGradient`
    pub fn jacobian_blocks(
        &self,
        u: &[f64],
        p: &[f64],
    ) -> (Vec<usize>, BlockMatrix) {
        let nt = self.nu + self.np;
        let mut coo = CooMatrix::<f64>::new(nt, nt);
        let ne = self.elem_dofs_u.len();

        for e in 0..ne {
            let et = self.mesh.element_type(e as u32);
            let ru_ref = et.ref_elem(self.order);
            let rp_ref = et.ref_elem(self.p_order);
            let n_du = ru_ref.n_dofs();
            let n_dp = rp_ref.n_dofs();
            let n_vd = n_du * self.dim;

            let eu = &self.elem_dofs_u[e];
            let ep = &self.elem_dofs_p[e];

            let mut ue = vec![0.0_f64; n_vd];
            for (k, &g) in eu.iter().enumerate() { ue[k] = u[g]; }
            let mut pe = vec![0.0_f64; n_dp];
            for (k, &g) in ep.iter().enumerate() { pe[k] = p[g]; }

            let q = ru_ref.quadrature(self.quad_order);
            let mut phi_u = vec![0.0_f64; n_du];
            let mut gr_u = vec![0.0_f64; n_du * self.dim];
            let mut gp_u = vec![0.0_f64; n_du * self.dim];
            let mut phi_p = vec![0.0_f64; n_dp];

            let mut kuu = vec![0.0_f64; n_vd * n_vd];
            let mut kup = vec![0.0_f64; n_vd * n_dp];
            let mut kpu = vec![0.0_f64; n_dp * n_vd];

            for (qi, xi) in q.points.iter().enumerate() {
                ru_ref.eval_basis(xi, &mut phi_u);
                ru_ref.eval_grad_basis(xi, &mut gr_u);
                rp_ref.eval_basis(xi, &mut phi_p);

                let (det_j, ji) = geometry_jacobian(&*self.mesh, e as u32, xi, self.dim);
                xform_grads(&ji, &gr_u, &mut gp_u, n_du, self.dim);
                // Quadrature weight and the geometric factor stay **separate**
                // below: MFEM's Jacobian closes every term with the tail chain
                // `* ip.weight * Tr.Weight()` (two roundings), not with a
                // pre-multiplied `w` (rr157 lane2, source (3)).

                let mut F = DMatrix::<f64>::identity(self.dim, self.dim);
                for k in 0..n_du {
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            F[(i, j)] += ue[k * self.dim + i] * gp_u[k * self.dim + j];
                        }
                    }
                }
                let dJ = mfem_det(&F);
                let FT = mfem_inverse_transpose(&F);

                let mut pres = 0.0;
                for k in 0..n_dp { pres += pe[k] * phi_p[k]; }

                // K_uu: MFEM IncompressibleNeoHookeanIntegrator::AssembleElementGrad
                // (nonlininteg.cpp:673-712) — three terms per qp, each closed by
                // the **tail chain** `* ip.weight * Tr.Weight()` (two separate
                // roundings, not a pre-multiplied `w`), accumulated with one
                // `+=` per (n,l) straight into the element matrix (n-outer /
                // l-inner exactly as C++):
                //   T1  dJ·(μF − pF⁻ᵀ)(i,l)·F⁻ᵀ(j,n)·DS_a(l)·DS_b(n)·ip.weight·Tr.Weight()
                //   T2  dJ·μ·DS_a(l)·DS_b(n)·ip.weight·Tr.Weight()   (i==j && n==l)
                //   T3  dJ·p·F⁻ᵀ(i,n)·F⁻ᵀ(j,l)·DS_a(l)·DS_b(n)·ip.weight·Tr.Weight()
                // (T1 was missing entirely and T2/T3 lacked dJ — the rr127
                // "solver structure debt" root cause, adjudicated rr148.)
                let wq = q.weights[qi];
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        for b in 0..n_du {
                            for j in 0..self.dim {
                                let col = b * self.dim + j;
                                for n in 0..self.dim {
                                    for l in 0..self.dim {
                                        kuu[row * n_vd + col] += dJ
                                            * (self.mu * F[(i, l)] - pres * FT[(i, l)])
                                            * FT[(j, n)]
                                            * gp_u[a * self.dim + l]
                                            * gp_u[b * self.dim + n]
                                            * wq
                                            * det_j;
                                        if i == j && n == l {
                                            kuu[row * n_vd + col] += dJ
                                                * self.mu
                                                * gp_u[a * self.dim + l]
                                                * gp_u[b * self.dim + n]
                                                * wq
                                                * det_j;
                                        }
                                        kuu[row * n_vd + col] += dJ
                                            * pres
                                            * FT[(i, n)]
                                            * FT[(j, l)]
                                            * gp_u[a * self.dim + l]
                                            * gp_u[b * self.dim + n]
                                            * wq
                                            * det_j;
                                    }
                                }
                            }
                        }
                    }
                }

                // K_up / K_pu: MFEM coupling blocks (nonlininteg.cpp:715-731) —
                // one per-l accumulated quantity
                // X = dJ·F⁻ᵀ(j,l)·DS_b(l)·φ_m·ip.weight·Tr.Weight(), added to
                // (1,0) and subtracted into (0,1).
                for m in 0..n_dp {
                    for b in 0..n_du {
                        for j in 0..self.dim {
                            let col = b * self.dim + j;
                            for l in 0..self.dim {
                                let x = dJ
                                    * FT[(j, l)]
                                    * gp_u[b * self.dim + l]
                                    * phi_p[m]
                                    * wq
                                    * det_j;
                                kpu[m * n_vd + col] += x;
                                kup[col * n_dp + m] -= x;
                            }
                        }
                    }
                }
            }

            for a in 0..n_vd { let gi = eu[a]; for b in 0..n_vd { coo.add(gi, eu[b], kuu[a * n_vd + b]); } }
            for a in 0..n_vd { let gi = eu[a]; for m in 0..n_dp { coo.add(gi, self.nu + ep[m], kup[a * n_dp + m]); } }
            for m in 0..n_dp { let gi = ep[m]; for b in 0..n_vd { coo.add(self.nu + gi, eu[b], kpu[m * n_vd + b]); } }
        }

        let mut flat = coo.into_csr();

        // Dirichlet row-zeroing
        let mut diag_scratch = vec![0.0_f64; nt];
        for &(dof, _) in &self.dirichlet {
            flat.apply_dirichlet_row_zeroing(dof, 0.0, &mut diag_scratch);
        }

        // Extract blocks
        let block_sizes = vec![self.nu, self.np];
        let mut bm = BlockMatrix::new_square(block_sizes.clone());

        let mut coo_uu = CooMatrix::new(self.nu, self.nu);
        for i in 0..self.nu {
            for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
                let c = flat.col_idx[p] as usize;
                if c < self.nu { coo_uu.add(i, c, flat.values[p]); }
            }
        }
        bm.set(0, 0, coo_uu.into_csr());

        let mut coo_up = CooMatrix::new(self.nu, self.np);
        for i in 0..self.nu {
            for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
                let c = flat.col_idx[p] as usize;
                if c >= self.nu && c < nt { coo_up.add(i, c - self.nu, flat.values[p]); }
            }
        }
        bm.set(0, 1, coo_up.into_csr());

        let mut coo_pu = CooMatrix::new(self.np, self.nu);
        for i in self.nu..nt {
            for p in flat.row_ptr[i]..flat.row_ptr[i + 1] {
                let c = flat.col_idx[p] as usize;
                if c < self.nu { coo_pu.add(i - self.nu, c, flat.values[p]); }
            }
        }
        bm.set(1, 0, coo_pu.into_csr());

        (block_sizes, bm)
    }

    /// Return the flattened CSR Jacobian (Dirichlet row-zeroed).
    ///
    /// Thin wrapper over [`Self::jacobian_blocks`] — the previous private
    /// re-assembly had drifted (missing T1/dJ plus node-dof scatter indices)
    /// and carried zero consumers; deleted in rr148 per the zero-dead-code
    /// discipline.
    pub fn jacobian_flat(&self, u: &[f64], p: &[f64]) -> CsrMatrix<f64> {
        let (_, bm) = self.jacobian_blocks(u, p);
        let nt = self.nu + self.np;
        let mut coo = CooMatrix::<f64>::new(nt, nt);
        let offsets = [0usize, self.nu];
        for (bi, &row_off) in offsets.iter().enumerate() {
            for (bj, &col_off) in offsets.iter().enumerate() {
                let Some(blk) = bm.get(bi, bj) else { continue };
                for r in 0..blk.nrows {
                    for p in blk.row_ptr[r]..blk.row_ptr[r + 1] {
                        coo.add(row_off + r, col_off + blk.col_idx[p] as usize, blk.values[p]);
                    }
                }
            }
        }
        let mut flat = coo.into_csr();
        let mut diag_scratch = vec![0.0_f64; nt];
        for &(dof, _) in &self.dirichlet {
            flat.apply_dirichlet_row_zeroing(dof, 0.0, &mut diag_scratch);
        }
        flat
    }
}
