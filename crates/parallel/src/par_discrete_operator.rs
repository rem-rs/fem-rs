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
        keep_owned_rows(&permuted, hcurl_part.n_owned_dofs, n_col_total)
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
        keep_owned_rows(&permuted, rt_part.n_owned_dofs, nd_part.n_total_dofs())
    }

    /// Build the parallel **transpose** of the discrete curl,
    /// `curlᵀ: H(div) → H(curl)` (3-D) — MFEM
    /// `NegCurl_->MultTranspose(HD_, RHS_)` in `MaxwellSolver::GetMaximumTimeStep`.
    ///
    /// Returns a `CsrMatrix` with:
    /// - rows = owned H(curl) DOFs on this rank
    /// - columns = total local H(div) DOFs (owned + ghost)
    ///
    /// This is the multi-rank-correct counterpart of taking
    /// [`Self::curl_3d`] and calling `.transpose()` at the call site.  That
    /// workaround transposes an *already ghost-row-truncated* matrix, so its
    /// columns are only this rank's owned H(div) DOFs: the resulting matrix
    /// cannot consume the local H(div) vector (length = owned + ghost) at all
    /// — `CsrMatrix::spmv` aborts on `x.len() == ncols` — and it silently drops
    /// every contribution of the ghost H(div) DOFs.  Here the serial curl is
    /// transposed *locally* (`H(div) × H(curl) → H(curl) × H(div)`), the result
    /// is permuted with the **H(curl)** partition as the row partition (so the
    /// by-DOF sign corrections of both spaces are applied exactly as in
    /// [`permute_rect_csr`]), and only the owned H(curl) rows are kept.
    ///
    /// The input vector must carry up-to-date ghost values
    /// (`ParVector::update_ghosts`): for an owned H(curl) DOF `p`, every H(div)
    /// DOF `i` with `curl[i][p] != 0` belongs to a local element, so summing
    /// over the *local* (owned + ghost) H(div) DOFs reproduces the exact global
    /// `(Cᵀ h)_p`.  At one rank (identity permutation, no ghost DOFs) this is
    /// bit-for-bit `Self::curl_3d(nd_par, rt_par).transpose()`.
    pub fn curl_3d_transpose(
        nd_par: &ParallelFESpace<HCurlSpace<fem_mesh::Mesh<3>>>,
        rt_par: &ParallelFESpace<HDivSpace<fem_mesh::Mesh<3>>>,
    ) -> CsrMatrix<f64> {
        let local_curl = DiscreteLinearOperator::curl_3d(
            nd_par.local_space(),
            rt_par.local_space(),
        ).expect("ParDiscreteLinearOperator::curl_3d_transpose: serial assembly failed");

        // H(div) rows × H(curl) columns → H(curl) rows × H(div) columns.  The
        // local transpose keeps every stored (explicit-zero included) entry, so
        // at one rank the result below is bit-identical to `curl_3d(..).transpose()`.
        let local_t = local_curl.transpose();

        let nd_part = nd_par.dof_partition();
        let rt_part = rt_par.dof_partition();
        let needs_perm = nd_part.needs_permutation() || rt_part.needs_permutation();
        let permuted = if needs_perm {
            permute_rect_csr(&local_t, nd_part, rt_part)
        } else {
            local_t
        };

        // Keep only owned H(curl) rows, all local H(div) columns.
        keep_owned_rows(&permuted, nd_part.n_owned_dofs, rt_part.n_total_dofs())
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
        keep_owned_rows(&permuted, l2_part.n_owned_dofs, rt_part.n_total_dofs())
    }
}

/// Keep the first `n_owned_rows` rows of a permuted rectangular CSR matrix,
/// dropping the ghost rows and any column outside `n_total_cols`.
///
/// That is the [owned rows | ghost columns] truncation of every parallel
/// discrete operator above: the ghost rows are a rank-local artefact, while the
/// ghost columns are what makes the matrix consumable by a local
/// (owned + ghost) vector.
fn keep_owned_rows(
    permuted: &CsrMatrix<f64>,
    n_owned_rows: usize,
    n_total_cols: usize,
) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::<f64>::new(n_owned_rows, n_total_cols);
    for row in 0..n_owned_rows.min(permuted.nrows) {
        for k in permuted.row_ptr[row]..permuted.row_ptr[row + 1] {
            let col = permuted.col_idx[k] as usize;
            let val = permuted.values[k];
            if val != 0.0 && col < n_total_cols {
                coo.add(row, col, val);
            }
        }
    }
    coo.into_csr()
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

    /// D108, single-rank limit: `curl_3d_transpose` must be bit-for-bit
    /// `curl_3d(..).transpose()` — the identity permutation and the absence of
    /// ghost DOFs make the two expressions the same matrix, so this pins that
    /// the local transposition, the row-space choice and the truncation are all
    /// neutral at one rank (the maxwell `--ranks 1` reference is unaffected).
    #[test]
    fn curl_3d_transpose_equals_curl_3d_transpose_at_one_rank() {
        use crate::dof_partition::DofPartition;
        use crate::launcher::native::ThreadLauncher;
        use crate::launcher::WorkerConfig;
        use crate::par_partition::partition_mesh;
        use crate::par_space::ParallelFESpace;
        use fem_mesh::Mesh;
        use fem_space::fe_space::FESpace as _;
        use fem_space::{HCurlSpace, HDivSpace};

        let mesh = Mesh::<3>::unit_cube_hex(2);
        ThreadLauncher::new(WorkerConfig::new(1)).launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let nd_par = ParallelFESpace::new_for_edge_space(
                HCurlSpace::new(lm.clone(), 1),
                &pmesh,
                comm.clone(),
            );
            let rt_local = HDivSpace::new(lm, 0);
            let rt_part = DofPartition::from_face_space(&rt_local, pmesh.partition(), &comm);
            let rt_par =
                ParallelFESpace::new_with_dof_partition(rt_local, rt_part, comm.clone());

            let ct = ParDiscreteLinearOperator::curl_3d_transpose(&nd_par, &rt_par);
            let old = ParDiscreteLinearOperator::curl_3d(&nd_par, &rt_par).transpose();

            assert_eq!(ct.nrows, old.nrows);
            assert_eq!(ct.ncols, old.ncols);
            for row in 0..ct.nrows {
                for col in 0..ct.ncols {
                    let (a, b) = (ct.get(row, col), old.get(row, col));
                    assert!(
                        a.to_bits() == b.to_bits(),
                        "rank {}: curl_3d_transpose[{row},{col}] = {a} vs \
                         curl_3d(..).transpose() = {b} (must be bit-identical)",
                        comm.rank()
                    );
                }
            }

            // The same statement through the operator action (identical entries
            // in identical per-row order, so the sums must agree to the bit).
            let x: Vec<f64> = (0..ct.ncols)
                .map(|i| ((i * 37 % 11) as f64) - 5.0)
                .collect();
            let mut y_new = vec![0.0f64; ct.nrows];
            let mut y_old = vec![0.0f64; old.nrows];
            ct.spmv(&x, &mut y_new);
            old.spmv(&x, &mut y_old);
            for (a, b) in y_new.iter().zip(y_old.iter()) {
                assert!(a.to_bits() == b.to_bits(), "spmv differs: {a} vs {b}");
            }
        });
    }

    /// D108, multi-rank: the owned-H(curl)-row transpose must be the rank-local
    /// transposed assembly permuted into the partition basis and truncated to
    /// the owned H(curl) rows —
    /// `par[permute(d)][permute(c)] == curl_locᵀ[d][c]·s_nd(d)·s_rt(c)` (the same
    /// reference and sign convention as the D88 test in `par_mixed_assembler`).
    ///
    /// Runs at 1, 2 and 4 ranks.  The shape assertions encode the D108 fix:
    /// `nrows` = owned H(curl) DOFs, `ncols` = *total local* H(div) DOFs, so the
    /// matrix can consume a local (owned + ghost) H(div) vector.
    #[test]
    fn curl_3d_transpose_is_owned_hcurl_rows_of_local_transpose() {
        use crate::dof_partition::DofPartition;
        use crate::launcher::native::ThreadLauncher;
        use crate::launcher::WorkerConfig;
        use crate::par_partition::partition_mesh;
        use crate::par_space::ParallelFESpace;
        use fem_mesh::Mesh;
        use fem_space::fe_space::FESpace as _;
        use fem_space::{HCurlSpace, HDivSpace};
        use std::sync::{Arc, Mutex};

        let mesh = Mesh::<3>::unit_cube_hex(2);
        for n_ranks in [1usize, 2, 4] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let seen_rank = Arc::clone(&seen);
            let mesh = mesh.clone();
            ThreadLauncher::new(WorkerConfig::new(n_ranks)).launch(move |comm| {
                let pmesh = partition_mesh(&mesh, &comm);
                let lm = pmesh.local_mesh().clone();
                let nd_par = ParallelFESpace::new_for_edge_space(
                    HCurlSpace::new(lm.clone(), 1),
                    &pmesh,
                    comm.clone(),
                );
                let rt_local = HDivSpace::new(lm, 0);
                let rt_part =
                    DofPartition::from_face_space(&rt_local, pmesh.partition(), &comm);
                let rt_par =
                    ParallelFESpace::new_with_dof_partition(rt_local, rt_part, comm.clone());

                let ct = ParDiscreteLinearOperator::curl_3d_transpose(&nd_par, &rt_par);
                let loc_t =
                    DiscreteLinearOperator::curl_3d(nd_par.local_space(), rt_par.local_space())
                        .expect("serial curl_3d")
                        .transpose();

                let ndp = nd_par.dof_partition();
                let rtp = rt_par.dof_partition();
                assert_eq!(ct.nrows, ndp.n_owned_dofs, "rows = owned H(curl) DOFs");
                assert_eq!(
                    ct.ncols,
                    rtp.n_total_dofs(),
                    "columns = local H(div) DOFs (owned + ghost)"
                );

                let n_nd_local = nd_par.local_space().n_dofs();
                let n_rt_local = rt_par.local_space().n_dofs();
                assert_eq!(loc_t.nrows, n_nd_local);
                assert_eq!(loc_t.ncols, n_rt_local);

                let mut max_dev = 0.0_f64;
                let mut checked_rows = 0usize;
                for d in 0..n_nd_local as u32 {
                    let p = ndp.permute_dof(d) as usize;
                    if p >= ndp.n_owned_dofs {
                        continue; // ghost H(curl) row — deliberately dropped
                    }
                    checked_rows += 1;
                    let sr = ndp.sign_correction(d);
                    for c in 0..n_rt_local as u32 {
                        let q = rtp.permute_dof(c) as usize;
                        let want =
                            loc_t.get(d as usize, c as usize) * sr * rtp.sign_correction(c);
                        let got = ct.get(p, q);
                        let dev = (want - got).abs();
                        if dev.is_nan() || !got.is_finite() {
                            max_dev = f64::INFINITY;
                        } else {
                            max_dev = max_dev.max(dev);
                        }
                    }
                }
                assert!(checked_rows > 0, "rank {}: no owned H(curl) rows", comm.rank());
                assert!(
                    max_dev < 1e-12,
                    "rank {} ({} ranks): owned-H(curl)-row curl transpose vs local \
                     transposed assembly max dev {max_dev:.3e}",
                    comm.rank(),
                    comm.size()
                );
                seen_rank.lock().unwrap().push((comm.rank(), checked_rows));
            });
            assert_eq!(
                seen.lock().unwrap().len(),
                n_ranks,
                "{n_ranks}-rank launch must visit every rank"
            );
        }
    }

    /// D108 gap pin (the reason the entry point exists rather than
    /// `.transpose()` at the call site): at 2 ranks the workaround transposes
    /// the already ghost-row-truncated H(div)-row matrix, so its columns are
    /// only the *owned* H(div) DOFs — strictly fewer than the local H(div)
    /// vector's length (= owned + ghost).  `CsrMatrix::spmv` cannot consume that
    /// vector at all, which is exactly the `left: 11520 / right: 5888` abort in
    /// `maxwell --ranks 2`.
    #[test]
    fn old_transpose_workaround_is_too_narrow_for_the_local_hdiv_vector() {
        use crate::dof_partition::DofPartition;
        use crate::launcher::native::ThreadLauncher;
        use crate::launcher::WorkerConfig;
        use crate::par_partition::partition_mesh;
        use crate::par_space::ParallelFESpace;
        use fem_mesh::Mesh;
        use fem_space::{HCurlSpace, HDivSpace};

        let mesh = Mesh::<3>::unit_cube_hex(2);
        ThreadLauncher::new(WorkerConfig::new(2)).launch(move |comm| {
            let pmesh = partition_mesh(&mesh, &comm);
            let lm = pmesh.local_mesh().clone();
            let nd = ParallelFESpace::new_for_edge_space(
                HCurlSpace::new(lm.clone(), 1),
                &pmesh,
                comm.clone(),
            );
            let rt_local = HDivSpace::new(lm, 0);
            let rt_part = DofPartition::from_face_space(&rt_local, pmesh.partition(), &comm);
            let rt =
                ParallelFESpace::new_with_dof_partition(rt_local, rt_part, comm.clone());

            let old = ParDiscreteLinearOperator::curl_3d(&nd, &rt).transpose();
            let new = ParDiscreteLinearOperator::curl_3d_transpose(&nd, &rt);

            let ndp = nd.dof_partition();
            let rtp = rt.dof_partition();
            assert!(rtp.n_ghost_dofs > 0, "expected ghost H(div) DOFs at 2 ranks");
            assert!(ndp.n_ghost_dofs > 0, "expected ghost H(curl) DOFs at 2 ranks");
            assert_eq!(
                old.ncols,
                rtp.n_owned_dofs,
                "the transpose workaround keeps only owned H(div) columns"
            );
            assert_ne!(
                old.ncols,
                rtp.n_total_dofs(),
                "-> it cannot consume the local H(div) vector (spmv length check)"
            );
            assert_eq!(
                old.nrows,
                ndp.n_total_dofs(),
                "the transpose workaround keeps ghost H(curl) rows too"
            );
            assert!(old.nrows > new.nrows, "the two paths must differ in shape");
            assert_eq!(new.nrows, ndp.n_owned_dofs, "owned H(curl) rows");
            assert_eq!(new.ncols, rtp.n_total_dofs(), "all local H(div) columns");
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
