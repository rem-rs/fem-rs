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
                let w = q.weights[qi] * det_j.abs();

                // Deformation gradient F = I + ∇u
                let mut F = DMatrix::<f64>::identity(self.dim, self.dim);
                for k in 0..n_du {
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            F[(i, j)] += ue[k * self.dim + i] * gp_u[k * self.dim + j];
                        }
                    }
                }
                let dJ = F.determinant();
                let iF = F.clone().try_inverse()
                    .unwrap_or_else(|| DMatrix::<f64>::identity(self.dim, self.dim));
                let FT = iF.transpose();

                let mut pres = 0.0;
                for k in 0..n_dp { pres += pe[k] * phi_p[k]; }

                // First Piola-Kirchhoff stress: P = μ·F - p·F^{-T}
                let mut P = DMatrix::<f64>::zeros(self.dim, self.dim);
                for i in 0..self.dim {
                    for j in 0..self.dim {
                        P[(i, j)] = self.mu * F[(i, j)] - pres * FT[(i, j)];
                    }
                }

                // R_u: P : ∇v
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        let mut s = 0.0;
                        for j in 0..self.dim {
                            s += P[(i, j)] * gp_u[a * self.dim + j];
                        }
                        fu_e[row] += w * s;
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
                let w = q.weights[qi] * det_j.abs();

                let mut F = DMatrix::<f64>::identity(self.dim, self.dim);
                for k in 0..n_du {
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            F[(i, j)] += ue[k * self.dim + i] * gp_u[k * self.dim + j];
                        }
                    }
                }
                let dJ = F.determinant();
                let iF = F.try_inverse()
                    .unwrap_or_else(|| DMatrix::<f64>::identity(self.dim, self.dim));
                let FT = iF.transpose();

                let mut pres = 0.0;
                for k in 0..n_dp { pres += pe[k] * phi_p[k]; }

                // K_uu: C_{iIjJ} = μ·δ_{ij}·δ_{IJ} + p·F^{-T}_{jI}·F^{-T}_{iJ}
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        for b in 0..n_du {
                            for j in 0..self.dim {
                                let col = b * self.dim + j;

                                let mut v = 0.0;
                                if i == j {
                                    for l in 0..self.dim {
                                        v += self.mu * gp_u[a * self.dim + l] * gp_u[b * self.dim + l];
                                    }
                                }
                                let ftn: f64 = (0..self.dim).map(|l| FT[(i, l)] * gp_u[b * self.dim + l]).sum();
                                let ftl: f64 = (0..self.dim).map(|l| FT[(j, l)] * gp_u[a * self.dim + l]).sum();
                                v += pres * ftn * ftl;

                                kuu[row * n_vd + col] += v * w;
                            }
                        }
                    }
                }

                // K_up: -∫ ψ_m · F^{-T}_{iJ} · ∂φ_a/∂X_J
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        let ft_gp: f64 = (0..self.dim).map(|l| FT[(i, l)] * gp_u[a * self.dim + l]).sum();
                        for m in 0..n_dp {
                            kup[row * n_dp + m] -= w * ft_gp * phi_p[m];
                        }
                    }
                }

                // K_pu: ∫ J · F^{-T}_{jJ} · ∂φ_b/∂X_J · ψ_m
                for m in 0..n_dp {
                    for b in 0..n_du {
                        for j in 0..self.dim {
                            let col = b * self.dim + j;
                            let ft_gp: f64 = (0..self.dim).map(|l| FT[(j, l)] * gp_u[b * self.dim + l]).sum();
                            kpu[m * n_vd + col] += w * dJ * ft_gp * phi_p[m];
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

    /// Return the flattened CSR matrix (for use with `NonlinearForm` trait).
    pub fn jacobian_flat(&self, u: &[f64], p: &[f64]) -> CsrMatrix<f64> {
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

            for (qi, xi) in q.points.iter().enumerate() {
                ru_ref.eval_basis(xi, &mut phi_u);
                ru_ref.eval_grad_basis(xi, &mut gr_u);
                rp_ref.eval_basis(xi, &mut phi_p);

                let (det_j, ji) = geometry_jacobian(&*self.mesh, e as u32, xi, self.dim);
                xform_grads(&ji, &gr_u, &mut gp_u, n_du, self.dim);
                let w = q.weights[qi] * det_j.abs();

                let mut F = DMatrix::<f64>::identity(self.dim, self.dim);
                for k in 0..n_du {
                    for i in 0..self.dim {
                        for j in 0..self.dim {
                            F[(i, j)] += ue[k * self.dim + i] * gp_u[k * self.dim + j];
                        }
                    }
                }
                let dJ = F.determinant();
                let iF = F.try_inverse()
                    .unwrap_or_else(|| DMatrix::<f64>::identity(self.dim, self.dim));
                let FT = iF.transpose();

                let mut pres = 0.0;
                for k in 0..n_dp { pres += pe[k] * phi_p[k]; }

                // K_uu
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        for b in 0..n_du {
                            for j in 0..self.dim {
                                let col = b * self.dim + j;
                                let mut v = 0.0;
                                if i == j {
                                    for l in 0..self.dim {
                                        v += self.mu * gp_u[a * self.dim + l] * gp_u[b * self.dim + l];
                                    }
                                }
                                let ftn: f64 = (0..self.dim).map(|l| FT[(i, l)] * gp_u[b * self.dim + l]).sum();
                                let ftl: f64 = (0..self.dim).map(|l| FT[(j, l)] * gp_u[a * self.dim + l]).sum();
                                v += pres * ftn * ftl;
                                coo.add(eu[a], eu[b], v * w);
                            }
                        }
                    }
                }

                // K_up
                for a in 0..n_du {
                    for i in 0..self.dim {
                        let row = a * self.dim + i;
                        let ft_gp: f64 = (0..self.dim).map(|l| FT[(i, l)] * gp_u[a * self.dim + l]).sum();
                        for m in 0..n_dp {
                            coo.add(eu[a], self.nu + ep[m], -w * ft_gp * phi_p[m]);
                        }
                    }
                }

                // K_pu
                for m in 0..n_dp {
                    for b in 0..n_du {
                        for j in 0..self.dim {
                            let ft_gp: f64 = (0..self.dim).map(|l| FT[(j, l)] * gp_u[b * self.dim + l]).sum();
                            coo.add(self.nu + ep[m], eu[b], w * dJ * ft_gp * phi_p[m]);
                        }
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
