//! Parallel nullspace (gradient) projector for singular H(curl) pencils.
//!
//! The discrete curl-curl operator `A` (after `EliminateEssentialBCDiag`)
//! has a large nullspace: every discrete gradient `Gφ` with `φ` constant on
//! each boundary component satisfies `A(Gφ) = 0`.  A plain LOBPCG
//! Rayleigh–Ritz then picks up these zero Ritz values and the projected
//! pencil becomes ill-conditioned (HYPRE AME's target problem — see
//! `vendor/linger/src/eigen/ame.rs`).
//!
//! [`ParGradientProjector`] applies the discrete divergence-free projection
//!
//! ```text
//! P = I − G (Gᵀ B G)⁻¹ Gᵀ B
//! ```
//!
//! (the parallel analog of HYPRE AME's `hypre_AMEDiscrDivFreeComponent`) to
//! a block of trial vectors, keeping the LOBPCG search space inside the
//! `B`-orthogonal complement of the gradient space.  The nodal operator
//! `GᵀBG` is solved with PCG + a pre-built [`ParAmgHierarchy`] V-cycle
//! (rebuilt once, reused across all applications).

use fem_linalg::CsrMatrix;
use fem_mesh::MeshTopology;
use fem_solver::SolverConfig;

use crate::par_amg::{ParAmgConfig, ParAmgHierarchy};
use crate::par_csr::ParCsrMatrix;
use crate::par_space::ParallelFESpace;
use crate::par_vector::ParVector;
use fem_space::H1Space;

/// Block projector `P = I − G(GᵀBG)⁻¹GᵀB` for singular H(curl) pencils.
///
/// `G` is the discrete gradient `H¹ → H(curl)` (owned rows × local H¹
/// columns, partition ordering), `B` the pencil mass matrix (a
/// [`ParCsrMatrix`]) and `GᵀBG` its pre-assembled nodal (H¹ Laplacian)
/// operator.
pub struct ParGradientProjector<'a, M: MeshTopology> {
    /// Discrete gradient: owned H(curl) rows × local H¹ columns.
    g: CsrMatrix<f64>,
    /// Pencil mass matrix `B` (H(curl)).
    b: &'a ParCsrMatrix,
    /// Nodal operator `GᵀBG` (H¹ Laplacian), parallel.
    nodal: &'a ParCsrMatrix,
    /// Pre-built AMG hierarchy for the nodal solve (built once).
    nodal_hierarchy: ParAmgHierarchy,
    /// PCG config for the nodal solve.
    nodal_cfg: SolverConfig,
    /// H¹ parallel space (layout + ghost exchange for the reverse step).
    h1: &'a ParallelFESpace<H1Space<M>>,
    n_owned_nd: usize,
}

impl<'a, M: MeshTopology> ParGradientProjector<'a, M> {
    /// Build the projector.
    ///
    /// * `h1` — parallel H¹ space (must share the local mesh with `nd`).
    /// * `g` — discrete gradient from
    ///   [`crate::ParDiscreteLinearOperator::gradient`].
    /// * `b` — the pencil mass matrix `B` (H(curl)).
    /// * `nodal` — `GᵀBG` assembled as the H¹ diffusion (Laplacian) operator.
    /// * `amg_cfg` — AMG configuration for the nodal solve.
    pub fn new(
        h1: &'a ParallelFESpace<H1Space<M>>,
        g: &CsrMatrix<f64>,
        b: &'a ParCsrMatrix,
        nodal: &'a ParCsrMatrix,
        amg_cfg: ParAmgConfig,
    ) -> Self {
        let comm = b.comm().clone();
        let nodal_hierarchy = ParAmgHierarchy::build(nodal, &comm, amg_cfg);
        ParGradientProjector {
            g: g.clone(),
            b,
            nodal,
            nodal_hierarchy,
            nodal_cfg: SolverConfig {
                rtol: 1e-10,
                atol: 0.0,
                max_iter: 200,
                verbose: false,
                ..SolverConfig::default()
            },
            h1,
            n_owned_nd: b.n_owned(),
        }
    }

    /// Apply `P` to a block of trial vectors in place.
    pub fn apply(&self, block: &mut [ParVector]) {
        let comm = self.b.comm();
        let n_owned_nd = self.n_owned_nd;
        let n_h1 = self.h1.dof_partition().n_total_dofs();
        let n_owned_h1 = self.h1.dof_partition().n_owned_dofs;
        let h1_exchange = self.h1.dof_ghost_exchange_arc();
        let g = &self.g;

        for v in block {
            // 1. w = B·v (owned H(curl)).
            let mut w = ParVector::zeros_like(v);
            self.b.spmv(v, &mut w);
            let wv = w.owned_slice();

            // 2. Gᵀw over local H¹ dofs (partition ordering, owned + ghost).
            let mut gtw = vec![0.0; n_h1];
            for i in 0..n_owned_nd {
                for k in g.row_ptr[i]..g.row_ptr[i + 1] {
                    gtw[g.col_idx[k] as usize] += g.values[k] * wv[i];
                }
            }

            // 3. Reverse ghost exchange: accumulate ghost-slot contributions
            //    back into the owning rank's owned slot.
            h1_exchange.reverse(comm, &mut gtw);

            // 4. Solve the nodal system (GᵀBG) u = GᵀBv with PCG + AMG.
            let rhs = ParVector::from_local_raw(gtw, n_owned_h1, h1_exchange.clone(), comm.clone());
            let mut u = ParVector::zeros(&self.h1);
            pcg_with_hierarchy(
                &self.nodal,
                &self.nodal_hierarchy,
                &rhs,
                &mut u,
                &self.nodal_cfg,
            );
            u.update_ghosts();
            let us = u.as_slice();

            // 5. v ← v − G·u (owned H(curl)).
            {
                let vs = v.owned_slice_mut();
                for i in 0..n_owned_nd {
                    let mut acc = 0.0;
                    for k in g.row_ptr[i]..g.row_ptr[i + 1] {
                        acc += g.values[k] * us[g.col_idx[k] as usize];
                    }
                    if acc != 0.0 {
                        vs[i] -= acc;
                    }
                }
            }
        }
    }
}

/// PCG with a pre-built AMG hierarchy as preconditioner (the body of
/// [`crate::par_solve_pcg_amg`] with the hierarchy built once).
fn pcg_with_hierarchy(
    a: &ParCsrMatrix,
    hierarchy: &ParAmgHierarchy,
    b: &ParVector,
    x: &mut ParVector,
    cfg: &SolverConfig,
) {
    let n = a.n_owned();
    let mut r = b.clone_vec();
    let mut ax = ParVector::zeros_like(b);
    a.spmv(&mut x.clone_vec(), &mut ax);
    for i in 0..n {
        r.data[i] = b.data[i] - ax.data[i];
    }

    let mut z = ParVector::zeros_like(b);
    hierarchy.vcycle(&r, &mut z);

    let mut p = z.clone_vec();
    let mut rz = r.global_dot(&z);
    let b_norm = b.global_norm();
    if b_norm < 1e-30 {
        return;
    }

    let mut ap = ParVector::zeros_like(b);
    for _ in 0..cfg.max_iter {
        a.spmv(&mut p, &mut ap);
        let pap = p.global_dot(&ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;

        x.axpy(alpha, &p);
        r.axpy(-alpha, &ap);

        if r.global_norm() / b_norm < cfg.rtol {
            return;
        }

        for v in z.as_slice_mut() {
            *v = 0.0;
        }
        hierarchy.vcycle(&r, &mut z);

        let rz_new = r.global_dot(&z);
        let beta = rz_new / rz;
        for i in 0..p.len() {
            p.data[i] = z.data[i] + beta * p.data[i];
        }
        rz = rz_new;
    }
}
