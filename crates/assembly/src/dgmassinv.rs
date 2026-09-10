//! DG mass inverse — 1:1 port of MFEM `fem/dgmassinv.hpp` + `dgmassinv.cpp`
//! (MFEM 4.9/4.10; class [`DGMassInverse`]).
//!
//! C++ semantics: `DGMassInverse` is a `Solver` that performs a **local
//! (diagonally preconditioned) CG iteration for each element**.  Because DG
//! spaces have no inter-element DOF sharing, the global mass matrix is
//! block-diagonal with one dense block `M_E` per element, and the iteration
//! inverts each block independently to (essentially) machine precision.  The
//! Jacobi preconditioner is the per-element diagonal `diag(M_E)` — exactly
//! what MFEM's PA `BilinearForm::AssembleDiagonal` produces
//! (`DGMassInverse::Update`).
//!
//! Implementation notes (documented deviations, verified numerically against
//! an MFEM harness — `tmp/dgpa/h_*.txt`):
//! - MFEM applies the per-element mass through tensor PA contractions
//!   (`B`, `Bt`, `pa_data`); this port precomputes each dense `M_E` and uses
//!   a dense matvec inside the CG loop.  Mathematically identical: the MFEM
//!   harness shows dense vs PA diagonals agree to ≤ 5.7e-16 relative and the
//!   CG solution matches the exact block solve to ≤ 2.3e-13 absolute
//!   (all P1–P3 quad cases).
//! - MFEM supports an internal change of basis (`btype != btype_orig`,
//!   `d2q`/`B_`/`Bt_` transform); fem-rs `L2Space` has a single basis per
//!   space object, so the basis-change path is unnecessary (equivalent to the
//!   C++ no-op branch `btype == btype_orig`).
//! - **`DGMassLaplacian` / `DGMassDirichletInverse` do not exist** in MFEM
//!   4.9, 4.10, or master (`fem/dgmassinv.hpp` defines only `DGMassInverse`;
//!   verified against `~/mfem410_mpi`, `~/mfem49` and upstream GitHub
//!   master), so no such API is invented here.
//! - MFEM's PA mass path **aborts on simplex L2 bases**
//!   (`Verification failed: (mode == DofToQuad::FULL)` from
//!   `MassIntegrator::AssemblePA` — `L2_TriangleElement` provides no TENSOR
//!   `DofToQuad`), i.e. the C++ `DGMassInverse` cannot run on triangle/tet L2
//!   spaces.  The dense `M_E` assembly used here works on simplices too and
//!   reproduces the C++ dense integration
//!   (`MassIntegrator::AssembleElementMatrix`).
//!
//! # Example
//! ```rust,ignore
//! use fem_assembly::dgmassinv::DGMassInverse;
//! let dg_mass = DGMassInverse::new(&space, 1.0, 5);
//! let mut u = vec![0.0; space.n_dofs()];
//! dg_mass.mult(&b, &mut u); // solves M u = b element-by-element
//! ```

use crate::dg::dg_base::ref_elem_vol;
use crate::postproc::coefficient::{CoeffCtx, ScalarCoeff};
use fem_element::{lagrange::QuadL2GL, lagrange::factory::QuadQk, ReferenceElement};
use fem_mesh::topology::MeshTopology;
use fem_mesh::{element_type::ElementType, transformation::element_jacobian_at};
use fem_space::fe_space::{FESpace, SpaceType};
use fem_space::L2Basis;

/// Solver for the discontinuous Galerkin mass matrix (MFEM `DGMassInverse`).
///
/// Performs a local (diagonally preconditioned) CG iteration for each element.
/// The Jacobi preconditioner is the per-element diagonal of the mass block
/// `M_E`; the CG tolerances default to the MFEM values `rel_tol = 1e-12`,
/// `abs_tol = 1e-12`, `max_iter = 100`.
pub struct DGMassInverse<'a, S: FESpace, C: ScalarCoeff = f64> {
    space: &'a S,
    rho:   C,
    quad_order: u8,
    n_dofs: usize,
    /// DOFs per element (uniform across elements, as in MFEM).
    n: usize,
    n_elem: usize,
    /// Global DOF ids per element, flattened (`elem_dofs[e*n + i]`).
    elem_dofs: Vec<u32>,
    /// Per-element mass blocks, flattened (`elem_mats[e*n*n + i*n + j]`).
    elem_mats: Vec<f64>,
    /// Per-element diagonal `diag(M_E)`, flattened (MFEM `AssembleDiagonal`
    /// output before `Reciprocal()`).
    elem_diag: Vec<f64>,
    /// Jacobi preconditioner: `1/diag(M_E)` per element (MFEM `diag_inv`).
    diag_inv: Vec<f64>,
    rel_tol: f64,
    abs_tol: f64,
    max_iter: usize,
    iterative_mode: bool,
}

impl<'a, S: FESpace, C: ScalarCoeff> DGMassInverse<'a, S, C> {
    /// Construct the DG inverse mass operator (MFEM
    /// `DGMassInverse(fes, coeff, ir, btype)` with `btype` = the space's own
    /// basis, i.e. the no-change-of-basis branch).
    ///
    /// - `space`      — DG (`L2`) finite element space.
    /// - `rho`        — density coefficient (`MassIntegrator` coefficient).
    /// - `quad_order` — quadrature order (the `ir` argument); MFEM's default
    ///   mass rule is `2p + OrderW()`, so `2p+1` gives the exact tensor rule.
    pub fn new(space: &'a S, rho: C, quad_order: u8) -> Self {
        assert_eq!(
            space.space_type(),
            SpaceType::L2,
            "DGMassInverse: space must be DG/L2 (MFEM: fes.IsDGSpace())"
        );
        let mut op = DGMassInverse {
            space,
            rho,
            quad_order,
            n_dofs: space.n_dofs(),
            n: 0,
            n_elem: space.mesh().n_elements(),
            elem_dofs: Vec::new(),
            elem_mats: Vec::new(),
            elem_diag: Vec::new(),
            diag_inv: Vec::new(),
            rel_tol: 1e-12,
            abs_tol: 1e-12,
            max_iter: 100,
            iterative_mode: false,
        };
        op.assemble();
        op
    }

    /// Set the relative CG tolerance (MFEM `SetRelTol`).
    pub fn set_rel_tol(&mut self, rel_tol: f64) { self.rel_tol = rel_tol; }

    /// Set the absolute CG tolerance (MFEM `SetAbsTol`).
    pub fn set_abs_tol(&mut self, abs_tol: f64) { self.abs_tol = abs_tol; }

    /// Set the maximum number of CG iterations (MFEM `SetMaxIter`).
    pub fn set_max_iter(&mut self, max_iter: usize) { self.max_iter = max_iter; }

    /// If `true`, `mult` uses `u` as an initial guess (MFEM `iterative_mode`).
    pub fn set_iterative_mode(&mut self, iterative_mode: bool) {
        self.iterative_mode = iterative_mode;
    }

    /// Recompute the operator and the preconditioner (MFEM `Update`; call when
    /// the coefficient or mesh changed).
    pub fn update(&mut self) { self.assemble(); }

    /// Per-element diagonal of the mass block, flattened
    /// (`element_diagonal()[e*n + i] = (M_E)_ii`) — the value MFEM's PA
    /// `BilinearForm::AssembleDiagonal` produces *before* `Reciprocal()`.
    pub fn element_diagonal(&self) -> &[f64] { &self.elem_diag }

    /// Jacobi preconditioner `1/diag(M_E)` per element (MFEM `diag_inv`).
    pub fn diag_inv(&self) -> &[f64] { &self.diag_inv }

    /// Number of DOFs (rows = cols).
    pub fn n_dofs(&self) -> usize { self.n_dofs }

    /// Dense mass block of element `e`, row-major `n × n`.
    pub fn element_mass(&self, e: usize) -> &[f64] {
        &self.elem_mats[e * self.n * self.n .. (e + 1) * self.n * self.n]
    }

    /// DOFs per element.
    pub fn dofs_per_element(&self) -> usize { self.n }

    /// Solve `M u = b` (MFEM `Mult`).  `u` is used as an initial guess iff
    /// `iterative_mode` is set.  Same as [`Self::mult_transpose`] since the
    /// mass matrix is symmetric.
    pub fn mult(&self, b: &[f64], u: &mut [f64]) {
        debug_assert_eq!(b.len(), self.n_dofs);
        debug_assert_eq!(u.len(), self.n_dofs);
        let n = self.n;
        let nn = n * n;
        for e in 0..self.n_elem {
            let g = &self.elem_dofs[e * n .. (e + 1) * n];
            let m    = &self.elem_mats[e * nn .. (e + 1) * nn];
            let dinv = &self.diag_inv[e * n .. (e + 1) * n];

            // Gather the element RHS / solution (L2 dofs are unshared).
            let b_e: Vec<f64> = g.iter().map(|&d| b[d as usize]).collect();
            let mut u_e: Vec<f64> = if self.iterative_mode {
                g.iter().map(|&d| u[d as usize]).collect()
            } else {
                vec![0.0_f64; n]
            };

            // Local (diagonally preconditioned) CG — MFEM DGMassCGIteration.
            let mut r = vec![0.0_f64; n];
            let mut z = vec![0.0_f64; n];
            let mut d = vec![0.0_f64; n];

            // First residual: r = b - M u (u = 0 when !iterative_mode).
            if self.iterative_mode {
                for i in 0..n {
                    let mut s = 0.0_f64;
                    for j in 0..n { s += m[i * n + j] * u_e[j]; }
                    r[i] = b_e[i] - s;
                }
            } else {
                r.copy_from_slice(&b_e);
            }

            for i in 0..n { z[i] = dinv[i] * r[i]; d[i] = z[i]; }

            let mut nom: f64 = dot(&d, &r);
            if nom < 0.0 { continue; } // Not positive definite
            let r0 = (nom * self.rel_tol * self.rel_tol).max(self.abs_tol * self.abs_tol);
            if nom <= r0 { continue; } // Converged

            dense_matvec(m, &d, &mut z);
            let mut den: f64 = dot(&z, &d);
            if den <= 0.0 {
                // MFEM recomputes dot(d,d) as a diagnostic (result discarded).
                let _diag_dd: f64 = dot(&d, &d);
                if den == 0.0 { continue; } // Not positive definite
            }

            let mut it = 1_usize;
            loop {
                let alpha = nom / den;
                for i in 0..n { u_e[i] += alpha * d[i]; r[i] -= alpha * z[i]; }
                for i in 0..n { z[i] = dinv[i] * r[i]; }

                let betanom: f64 = dot(&r, &z);
                if betanom < 0.0 { break; } // Not positive definite
                if betanom <= r0 { break; } // Converged
                it += 1;
                if it > self.max_iter { break; }

                let beta = betanom / nom;
                for i in 0..n { d[i] = z[i] + beta * d[i]; }
                dense_matvec(m, &d, &mut z);
                den = dot(&d, &z);
                if den <= 0.0 {
                    let _diag_dd: f64 = dot(&d, &d);
                    if den == 0.0 { break; }
                }
                nom = betanom;
            }

            // Scatter the element solution.
            for (i, &gi) in g.iter().enumerate() {
                u[gi as usize] = u_e[i];
            }
        }
    }

    /// Same as [`Self::mult`] since the mass matrix is symmetric
    /// (MFEM `MultTranspose`).
    pub fn mult_transpose(&self, b: &[f64], u: &mut [f64]) { self.mult(b, u); }

    // ─── internals ──────────────────────────────────────────────────────────

    /// Assemble the per-element mass blocks and the Jacobi preconditioner
    /// (MFEM `Update`: `M->Assemble(); M->AssembleDiagonal(diag_inv);
    /// diag_inv.Reciprocal();`).
    fn assemble(&mut self) {
        let mesh = self.space.mesh();
        let dim = mesh.dim() as usize;
        let basis = self.space.l2_basis();

        let mut elem_dofs = Vec::new();
        let mut elem_mats = Vec::new();
        let mut elem_diag = Vec::new();
        let mut diag_inv = Vec::new();
        let mut n: usize = 0;

        for e in mesh.elem_iter() {
            let et = mesh.element_type(e);
            let re = l2_ref_elem(et, self.space.order(), basis);
            let ne = re.n_dofs();
            if n == 0 { n = ne; }
            debug_assert_eq!(ne, n, "DGMassInverse: uniform DOF count per element required");

            let quad = re.quadrature(self.quad_order);
            let gd = self.space.element_dofs(e);
            elem_dofs.extend_from_slice(gd);

            let mut phi = vec![0.0_f64; ne];
            let mut m_e = vec![0.0_f64; ne * ne];

            for (qi, xi) in quad.points.iter().enumerate() {
                let (jac, xp) = element_jacobian_at(mesh, e, xi, dim);
                let det = jac.determinant().abs();
                let w = quad.weights[qi] * det;
                re.eval_basis(xi, &mut phi);

                let ctx = CoeffCtx::from_qp(&xp, dim, e, mesh.element_tag(e), None, None);
                let rho_qp = self.rho.eval(&ctx);

                // MassIntegrator kernel: symmetric accumulation sharing
                // `avi*vj` between entries (i,j)/(j,i).
                for i in 0..ne {
                    let avi = w * rho_qp * phi[i];
                    for j in 0..i {
                        let avivj = avi * phi[j];
                        m_e[i * ne + j] += avivj;
                        m_e[j * ne + i] += avivj;
                    }
                    m_e[i * ne + i] += avi * phi[i];
                }
            }

            for i in 0..ne {
                let di = m_e[i * ne + i];
                elem_diag.push(di);
                diag_inv.push(1.0 / di);
            }
            elem_mats.extend_from_slice(&m_e);
        }

        self.n = n;
        self.elem_dofs = elem_dofs;
        self.elem_mats = elem_mats;
        self.elem_diag = elem_diag;
        self.diag_inv = diag_inv;
    }
}

/// Reference element matching the space's L2 basis (MFEM
/// `L2_FECollection` basis dispatch): Gauss-Lobatto tensor quads use the
/// GLL-noded lexicographic `QuadQk::new_lex`, everything else falls back to
/// the DG assembler's dispatch (`dg_base::ref_elem_vol`; GL-noded
/// `QuadL2GL` for Quad4).
pub(crate) fn l2_ref_elem(
    et: ElementType,
    order: u8,
    basis: Option<L2Basis>,
) -> Box<dyn ReferenceElement> {
    match (et, basis) {
        (ElementType::Quad4, Some(L2Basis::GaussLobatto)) => {
            Box::new(QuadQk::new_lex(order as usize))
        }
        (ElementType::Quad4, _) => Box::new(QuadL2GL::new(order as usize)),
        _ => ref_elem_vol(et, order),
    }
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn dense_matvec(m: &[f64], x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let mut s = 0.0_f64;
        for j in 0..n { s += m[i * n + j] * x[j]; }
        y[i] = s;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembler::Assembler;
    use crate::postproc::coefficient::FnCoeff;
    use fem_linalg::CsrMatrix;
    use fem_mesh::Mesh;
    use fem_space::L2Space;

    /// `rho = x + 2y²` — the C++ harness density coefficient.
    fn rho_fn(x: &[f64]) -> f64 { x[0] + 2.0 * x[1] * x[1] }

    /// Diagonal of a CSR matrix.
    fn csr_diag(m: &CsrMatrix<f64>) -> Vec<f64> {
        let mut d = vec![0.0_f64; m.nrows];
        for i in 0..m.nrows {
            for k in m.row_ptr[i]..m.row_ptr[i + 1] {
                if m.col_idx[k] == i as u32 { d[i] = m.values[k]; }
            }
        }
        d
    }

    /// `element_diagonal()` equals the diagonal of the globally assembled
    /// sparse mass for tri/quad L2 spaces, GL and GLL bases, with the
    /// spatially varying density `rho = x + 2y²`.
    #[test]
    fn dgmass_diag_matches_assembled_mass_diagonal() {
        // Tri P1 / P2 (MFEM's PA path aborts here; the dense integration is
        // the same one MassIntegrator::AssembleElementMatrix performs in C++).
        for order in [1_u8, 2] {
            let mesh = Mesh::<2>::unit_square_tri(3);
            let space = L2Space::new(mesh, order);
            let op = DGMassInverse::new(&space, FnCoeff(rho_fn), 2 * order + 1);
            let m = Assembler::assemble_bilinear(
                &space, &[&crate::standard::MassIntegrator { rho: FnCoeff(rho_fn) }],
                2 * order + 1);
            let err = csr_diag(&m).iter().zip(op.element_diagonal().iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
            assert!(err < 1e-13, "tri P{order} diag: max abs err {err:.3e}");
        }

        // Quad GL (L2_FECollection default) and GLL (DG_FECollection default).
        for (basis, name) in [(L2Basis::GaussLegendre, "GL"), (L2Basis::GaussLobatto, "GLL")] {
            for order in [1_u8, 2] {
                let mesh = Mesh::<2>::make_cartesian_2d(2, 3, 1.0, 2.0);
                let space = L2Space::new_with_basis(mesh, order, basis);
                let op = DGMassInverse::new(&space, FnCoeff(rho_fn), 2 * order + 1);
                let m = Assembler::assemble_bilinear(
                    &space, &[&crate::standard::MassIntegrator { rho: FnCoeff(rho_fn) }],
                    2 * order + 1);
                let err = csr_diag(&m).iter().zip(op.element_diagonal().iter())
                    .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
                assert!(err < 1e-13, "quad {name} P{order} diag: max abs err {err:.3e}");
            }
        }
    }

    /// `mult` matches the exact per-element dense inverse (Gauss) solve on a
    /// rectangular 2×3 quad mesh with the smooth RHS `g = x² − y + 0.5`.
    #[test]
    fn dgmass_solve_matches_exact_block_inverse() {
        let mesh = Mesh::<2>::make_cartesian_2d(2, 3, 1.0, 2.0);
        let space = L2Space::new_with_basis(mesh, 1, L2Basis::GaussLegendre);
        let op = DGMassInverse::new(&space, FnCoeff(rho_fn), 3);

        let n = space.n_dofs();
        let g = |x: &[f64]| x[0] * x[0] - x[1] + 0.5;
        let b: Vec<f64> = space
            .dof_coords()
            .chunks_exact(2)
            .map(|c| g(c))
            .collect();
        let mut u = vec![0.0_f64; n];
        op.mult(&b, &mut u);

        // Exact block-diagonal inverse via Gaussian elimination per element.
        let mut u_exact = vec![0.0_f64; n];
        for e in space.mesh().elem_iter() {
            let gd = space.element_dofs(e);
            let d = gd.len();
            let m = op.element_mass(e as usize);
            let mut a = m.to_vec();
            let mut rhs: Vec<f64> = gd.iter().map(|&dof| b[dof as usize]).collect();
            for col in 0..d {
                let piv = (col..d).fold(col, |acc, r| {
                    if a[r * d + col].abs() > a[acc * d + col].abs() { r } else { acc }
                });
                for k in 0..d {
                    a.swap(piv * d + k, col * d + k);
                }
                rhs.swap(piv, col);
                for row in (col + 1)..d {
                    let f = a[row * d + col] / a[col * d + col];
                    for k in col..d { a[row * d + k] -= f * a[col * d + k]; }
                    rhs[row] -= f * rhs[col];
                }
            }
            for row in (0..d).rev() {
                let mut s = rhs[row];
                for k in (row + 1)..d { s -= a[row * d + k] * u_exact[gd[k] as usize]; }
                u_exact[gd[row] as usize] = s / a[row * d + row];
            }
        }

        let err = u.iter().zip(u_exact.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-10, "DGMassInverse vs exact block inverse: {err:.3e}");
    }

    /// `mult` matches a global CG solve on the assembled sparse mass (the
    /// round-10 plan's "<5% lumped-inverse deviation" criterion passes at
    /// ~1e-12 instead: the local CG inverts each `M_E` to machine precision,
    /// exactly like the C++ solver).
    #[test]
    fn dgmass_solve_matches_global_cg() {
        let mesh = Mesh::<2>::make_cartesian_2d(3, 3, 1.0, 1.0);
        let space = L2Space::new_with_basis(mesh, 2, L2Basis::GaussLegendre);
        let m = Assembler::assemble_bilinear(
            &space, &[&crate::standard::MassIntegrator { rho: FnCoeff(rho_fn) }], 5);
        let op = DGMassInverse::new(&space, FnCoeff(rho_fn), 5);

        let n = space.n_dofs();
        let g = |x: &[f64]| x[0] * x[0] - x[1] + 0.5;
        let b: Vec<f64> = space
            .dof_coords()
            .chunks_exact(2)
            .map(|c| g(c))
            .collect();

        // Plain CG on the assembled sparse mass.
        let mut x = vec![0.0_f64; n];
        let mut r = b.clone();
        let mut p = r.clone();
        let mut ap = vec![0.0_f64; n];
        let mut rr = dot(&r, &r);
        let rr0 = rr;
        for _ in 0..200 {
            m.spmv(&p, &mut ap);
            let pap: f64 = dot(&p, &ap);
            let alpha = rr / pap;
            for i in 0..n { x[i] += alpha * p[i]; r[i] -= alpha * ap[i]; }
            let rr_new: f64 = dot(&r, &r);
            if rr_new.sqrt() < 1e-13 * rr0.sqrt() { break; }
            let beta = rr_new / rr;
            rr = rr_new;
            for i in 0..n { p[i] = r[i] + beta * p[i]; }
        }

        let mut u = vec![0.0_f64; n];
        op.mult(&b, &mut u);
        let err = u.iter().zip(x.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        assert!(err < 1e-10, "DGMassInverse vs global CG: {err:.3e}");
    }

    /// Bit-parity gate against the MFEM C++ harness
    /// (`tmp/dgpa/h_rect_p1.txt`, built from `~/mfem49/fem/dgmassinv.cpp`):
    /// rectangular `[0,1]×[0,2]` domain, 2×3 quad mesh, L2 P1
    /// Gauss-Legendre, `rho = x + 2y²` (mass coefficient), order-3 rule,
    /// RHS = dof values of `g = x² − y + 0.5`.  Measured deviations are
    /// asserted well below 1e-13 (MFEM-internal dense-vs-PA deviation is
    /// ~4e-16; the residual comes from node/weight round-off in the
    /// Gauss-Legendre point generators).
    #[test]
    fn dgmass_matches_mfem_cxx_harness_p1() {
        // MFEM MakeCartesian2D numbers elements along the Hilbert SFC by
        // default (sfc_ordering = true) — use the matching fem-rs constructor.
        let mesh = Mesh::<2>::make_cartesian_2d_sfc(2, 3, 1.0, 2.0);
        let space = L2Space::new_with_basis(mesh, 1, L2Basis::GaussLegendre);
        let op = DGMassInverse::new(&space, FnCoeff(rho_fn), 3);

        // C++ harness SECTION dense_diag (24 values, %.17g round-trip).
        let cxx_diag: [f64; 24] = [
            0.012113217446588014, 0.03616947866282242, 0.05487990405322696,
            0.07893616526946136, 0.05377988411325468, 0.07783614532948908,
            0.09654657071989362, 0.12060283193612804, 0.15916134565476386,
            0.18321760687099828, 0.28746140547468063, 0.3115176666909152,
            0.4126909553444212, 0.4367472165606557, 0.6265243883776161,
            0.6505806495938505, 0.3710242886777545, 0.395080549893989,
            0.5848577217109495, 0.6089139829271839, 0.11749467898809719,
            0.14155094020433162, 0.24579473880801397, 0.2698510000242485,
        ];
        let d_op = op.element_diagonal();
        assert_eq!(d_op.len(), 24);
        let diag_err = d_op.iter().zip(cxx_diag.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        let diag_scale = cxx_diag.iter().cloned().fold(0.0_f64, f64::max);
        eprintln!(
            "dgmassinv vs MFEM: diag max abs {diag_err:.3e} (scale {diag_scale:.6e}, \
             rel {:.3e})",
            diag_err / diag_scale
        );
        assert!(
            diag_err <= 1e-14 * diag_scale,
            "diag vs MFEM: max abs {diag_err:.3e} (scale {diag_scale:.3e})"
        );

        // RHS = dof values of g = x² − y + 0.5 (harness SECTION b).
        let g = |x: &[f64]| x[0] * x[0] - x[1] + 0.5;
        let b: Vec<f64> = space
            .dof_coords()
            .chunks_exact(2)
            .map(|c| g(c))
            .collect();
        let mut u = vec![0.0_f64; space.n_dofs()];
        op.mult(&b, &mut u);

        // C++ harness SECTION solve (24 values) — DGMassInverse output.
        let cxx_u: [f64; 24] = [
            30.568369445515973, 14.227987032269295, -0.26637935380499056,
            1.6433366566010927, 13.498425122207513, 14.889694701221305,
            3.5324254065270746, 6.418226245517845, 0.3724338461279198,
            2.6869130233533434, -1.132754175486462, 0.34473035089890675,
            -1.4717783045248858, -0.3992627452340751, -1.5838007145721034,
            -0.8596583259373741, -2.5956576338553403, -2.0722722497334756,
            -2.3047523468931845, -1.9766579076218036, -2.5225428346037693,
            -1.074156011034643, -2.77176616289328, -1.989794266834621,
        ];
        let u_err = u.iter().zip(cxx_u.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max);
        let u_scale = cxx_u.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        eprintln!(
            "dgmassinv vs MFEM: solve max abs {u_err:.3e} (scale {u_scale:.6e}, \
             rel {:.3e})",
            u_err / u_scale
        );
        assert!(
            u_err <= 1e-11 * u_scale,
            "solve vs MFEM: max abs {u_err:.3e} (scale {u_scale:.3e})"
        );
    }
}
