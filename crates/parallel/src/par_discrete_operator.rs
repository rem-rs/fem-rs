//! Parallel discrete linear operators.
//!
//! Provides [`ParDiscreteLinearOperator::gradient`] for building the discrete
//! gradient `G: H¹ → H(Curl)` in parallel, used by the AMS preconditioner,
//! and [`ParDiscreteLinearOperator::curl_3d`] for `curl: H(Curl) → H(div)`
//! (used by pex34 to recover `B = curl A`).
//!
//! Each rank assembles the operator on its local mesh (with ghost overlap),
//! permutes to the parallel DOF ordering, and returns a `CsrMatrix` with
//! owned-row × full-local-column structure.

use fem_linalg::{CooMatrix, CsrMatrix};
use fem_space::{H1Space, HCurlSpace, HDivSpace, L2Space};
use fem_assembly::DiscreteLinearOperator;
use fem_mesh::MeshTopology;

use crate::par_mixed_assembler::permute_rect_csr;
use crate::par_space::ParallelFESpace;

pub struct ParDiscreteLinearOperator;

impl ParDiscreteLinearOperator {
    /// Build the discrete gradient `G: H¹ → H(Curl)` in parallel.
    ///
    /// Returns a `CsrMatrix` with:
    /// - rows = owned H(Curl) DOFs on this rank
    /// - columns = total local H¹ DOFs (owned + ghost)
    ///
    /// Both parallel spaces must share the same local mesh (with ghost overlap).
    pub fn gradient(
        h1_par: &ParallelFESpace<H1Space<fem_mesh::Mesh<3>>>,
        hcurl_par: &ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
    ) -> CsrMatrix<f64> {
        // Serial gradient on the local mesh.
        let local_grad = DiscreteLinearOperator::gradient(
            h1_par.local_space(),
            hcurl_par.local_space(),
        ).expect("ParDiscreteLinearOperator::gradient: serial assembly failed");

        let h1_part = h1_par.dof_partition();
        let hcurl_part = hcurl_par.dof_partition();

        // Permute rows (HCur) and columns (H¹) to parallel ordering.
        let n_row_total = hcurl_part.n_total_dofs();
        let n_col_total = h1_part.n_total_dofs();
        let needs_perm = hcurl_part.needs_permutation() || h1_part.needs_permutation();

        let permuted = if needs_perm {
            let mut coo = CooMatrix::<f64>::new(n_row_total, n_col_total);
            for row in 0..local_grad.nrows {
                let new_row = hcurl_part.permute_dof(row as u32) as usize;
                for k in local_grad.row_ptr[row]..local_grad.row_ptr[row + 1] {
                    let col = local_grad.col_idx[k] as usize;
                    let new_col = h1_part.permute_dof(col as u32) as usize;
                    let val = local_grad.values[k];
                    if val != 0.0 {
                        coo.add(new_row, new_col, val);
                    }
                }
            }
            coo.into_csr()
        } else {
            local_grad
        };

        // Keep only owned rows (discard ghost rows).
        let n_owned_rows = hcurl_part.n_owned_dofs;
        let mut coo = CooMatrix::<f64>::new(n_owned_rows, n_col_total);
        for row in 0..n_owned_rows.min(permuted.nrows) {
            for k in permuted.row_ptr[row]..permuted.row_ptr[row + 1] {
                let col = permuted.col_idx[k] as usize;
                let val = permuted.values[k];
                if val != 0.0 && col < n_col_total {
                    coo.add(row, col, val);
                }
            }
        }
        coo.into_csr()
    }

    /// Build the discrete curl `curl: H(Curl) → H(div)` in parallel (3-D).
    ///
    /// Returns a `CsrMatrix` with:
    /// - rows = owned H(div) DOFs on this rank
    /// - columns = total local H(Curl) DOFs (owned + ghost)
    ///
    /// The serial `DiscreteLinearOperator::curl_3d` is assembled on the local
    /// mesh (with ghost overlap) and permuted with the per-DOF sign
    /// corrections of both spaces (same convention as
    /// [`crate::ParMixedAssembler::assemble_hcurl_hdiv_curl`]).
    pub fn curl_3d(
        nd_par: &ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
        rt_par: &ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    ) -> CsrMatrix<f64> {
        let local_curl = DiscreteLinearOperator::curl_3d(
            nd_par.local_space(),
            rt_par.local_space(),
        ).expect("ParDiscreteLinearOperator::curl_3d: serial assembly failed");

        let nd_part = nd_par.dof_partition();
        let rt_part = rt_par.dof_partition();
        let needs_perm = nd_part.needs_permutation() || rt_part.needs_permutation();
        let permuted = if needs_perm {
            permute_rect_csr(&local_curl, rt_part, nd_part)
        } else {
            local_curl
        };

        // Keep only owned H(div) rows.
        let n_owned_rows = rt_part.n_owned_dofs;
        let n_col_total = nd_part.n_total_dofs();
        let mut coo = CooMatrix::<f64>::new(n_owned_rows, n_col_total);
        for row in 0..n_owned_rows.min(permuted.nrows) {
            for k in permuted.row_ptr[row]..permuted.row_ptr[row + 1] {
                let col = permuted.col_idx[k] as usize;
                let val = permuted.values[k];
                if val != 0.0 && col < n_col_total {
                    coo.add(row, col, val);
                }
            }
        }
        coo.into_csr()
    }

    /// Build the discrete divergence `div: H(div) → L²` in parallel.
    ///
    /// Implemented by [`crate::assembly::discrete_op::DiscreteLinearOperator::divergence`],
    /// which for RT0 → P0 is the signed face-element incidence matrix
    /// (topological, exact) and for RT1 → P1/P2 and RT2 → P2 is the
    /// commuting-diagram interpolation div `Π_{L²} ∘ div` (DOF-functional
    /// projection on each reference element).  This is the operator used by
    /// MFEM's `ParDiscreteDivOperator` (volta/tesla `rho_ = div(D)`), distinct
    /// from the weak-div bilinear form `∫ p · div(v)`.
    ///
    /// Returns a `CsrMatrix` with:
    /// - rows = owned L² DOFs on this rank
    /// - columns = total local H(div) DOFs (owned + ghost)
    pub fn divergence<M: MeshTopology>(
        rt_par: &ParallelFESpace<HDivSpace<M>>,
        l2_par: &ParallelFESpace<L2Space<M>>,
    ) -> CsrMatrix<f64> {
        let local_div = DiscreteLinearOperator::divergence(
            rt_par.local_space(),
            l2_par.local_space(),
        ).expect("ParDiscreteLinearOperator::divergence: serial assembly failed");

        let rt_part = rt_par.dof_partition();
        let l2_part = l2_par.dof_partition();
        let needs_perm = rt_part.needs_permutation() || l2_part.needs_permutation();
        let permuted = if needs_perm {
            permute_rect_csr(&local_div, l2_part, rt_part)
        } else {
            local_div
        };

        // Keep only owned L² rows.
        let n_owned_rows = l2_part.n_owned_dofs;
        let n_col_total = rt_part.n_total_dofs();
        let mut coo = CooMatrix::<f64>::new(n_owned_rows, n_col_total);
        for row in 0..n_owned_rows.min(permuted.nrows) {
            for k in permuted.row_ptr[row]..permuted.row_ptr[row + 1] {
                let col = permuted.col_idx[k] as usize;
                let val = permuted.values[k];
                if val != 0.0 && col < n_col_total {
                    coo.add(row, col, val);
                }
            }
        }
        coo.into_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launcher::native::ThreadLauncher;
    use crate::launcher::WorkerConfig;
    use crate::par_partition::partition_mesh;
    use fem_linalg::CsrMatrix;
    use fem_mesh::Mesh;
    use fem_space::{HDivSpace, L2Space};

    /// Test: `ParDiscreteLinearOperator::divergence` dimensions on 2D RT0→P0.
    #[test]
    fn par_divergence_rt0_p0_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);
            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let d = ParDiscreteLinearOperator::divergence(&rt_par, &l2_par);
            assert_eq!(d.nrows, l2_par.dof_partition().n_owned_dofs);
            assert_eq!(d.ncols, rt_par.dof_partition().n_total_dofs());
            assert!(d.nnz() > 0, "divergence matrix should be non-empty");
        });
    }

    /// Test: multi-rank divergence dimensions — 2 ranks, 2D RT0→P0.
    #[test]
    fn par_divergence_rt0_p0_two_ranks_dimensions() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let launcher = ThreadLauncher::new(WorkerConfig::new(2));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);
            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let d = ParDiscreteLinearOperator::divergence(&rt_par, &l2_par);
            assert_eq!(d.nrows, l2_par.dof_partition().n_owned_dofs);
            assert_eq!(d.ncols, rt_par.dof_partition().n_total_dofs());
            assert!(d.nnz() > 0, "rank {}: divergence matrix should be non-empty", comm.rank());
        });
    }

    /// Test: 3D RT0→P0 divergence dimensions.
    #[test]
    fn par_divergence_rt0_p0_3d_dimensions() {
        let mesh = Mesh::<3>::unit_cube_tet(2);
        let launcher = ThreadLauncher::new(WorkerConfig::new(1));
        launcher.launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let rt_local = HDivSpace::new(pmesh.local_mesh().clone(), 0);
            let l2_local = L2Space::new(pmesh.local_mesh().clone(), 0);
            let rt_par = ParallelFESpace::new(rt_local, &pmesh, comm.clone());
            let l2_par = ParallelFESpace::new(l2_local, &pmesh, comm.clone());

            let d = ParDiscreteLinearOperator::divergence(&rt_par, &l2_par);
            assert_eq!(d.nrows, l2_par.dof_partition().n_owned_dofs);
            assert_eq!(d.ncols, rt_par.dof_partition().n_total_dofs());
            assert!(d.nnz() > 0, "divergence matrix should be non-empty");
        });
    }
}
