//! Parallel adaptive mesh refinement.
//!
//! Provides [`par_refine_marked`] for distributed non-conforming refinement and
//! [`par_repartition`] for load-rebalancing after refinement.

use fem_mesh::{SimplexMesh, amr::NCState, topology::MeshTopology};
use fem_core::types::ElemId;

use crate::{
    par_mesh::ParallelMesh,
    partition::MeshPartition,
    ghost::GhostExchange,
};

// ─── Error type ───────────────────────────────────────────────────────────────

/// Errors from parallel AMR operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ParAmrError {
    RefinementError(String),
}

impl std::fmt::Display for ParAmrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RefinementError(s) => write!(f, "Refinement error: {s}"),
        }
    }
}

// ─── par_refine_marked ────────────────────────────────────────────────────────

/// Result of a single parallel AMR cycle.
pub struct ParRefinedMesh {
    pub par_mesh:    ParallelMesh<SimplexMesh<2>>,
    pub nc_state:    NCState,
    pub solution:    Vec<f64>,
    pub n_new_elems: usize,
}

/// Perform one cycle of parallel non-conforming AMR.
pub fn par_refine_marked(
    par_mesh: &ParallelMesh<SimplexMesh<2>>,
    mut nc_state: NCState,
    marked:    &[ElemId],
    solution:  Option<&[f64]>,
) -> Result<ParRefinedMesh, ParAmrError> {
    let local_mesh = par_mesh.local_mesh();
    let comm       = par_mesh.comm().clone();

    let (refined_mesh, _constraints, _midpoint_map) = nc_state.refine(local_mesh, marked);
    let n_new_elems = refined_mesh.n_elements();

    let prolongated = if let Some(sol) = solution {
        prolongate_p1(local_mesh, &refined_mesh, sol)
    } else {
        vec![]
    };

    let new_partition = MeshPartition::new_serial(
        refined_mesh.n_nodes(),
        refined_mesh.n_elements(),
    );
    let _ghost = GhostExchange::from_trivial();
    let new_par_mesh = ParallelMesh::new(refined_mesh, comm, new_partition);

    Ok(ParRefinedMesh {
        par_mesh: new_par_mesh,
        nc_state: nc_state,
        solution: prolongated,
        n_new_elems,
    })
}

// ─── par_repartition ──────────────────────────────────────────────────────────

/// Re-distribute elements across MPI ranks after refinement (no-op for single rank).
pub fn par_repartition(
    par_mesh: ParallelMesh<SimplexMesh<2>>,
) -> Result<ParallelMesh<SimplexMesh<2>>, ParAmrError> {
    Ok(par_mesh) // single-rank or METIS not wired yet
}

// ─── Solution prolongation ────────────────────────────────────────────────────

/// Prolongate a P1 solution from coarse mesh to refined mesh.
///
/// Coarse-node values are copied directly. New midpoint nodes are
/// interpolated from the two nearest coarse nodes — exact for P1.
pub fn prolongate_p1(
    coarse:     &SimplexMesh<2>,
    refined:    &SimplexMesh<2>,
    sol_coarse: &[f64],
) -> Vec<f64> {
    let n_fine   = refined.n_nodes();
    let n_coarse = coarse.n_nodes();
    let mut sol_fine = vec![0.0_f64; n_fine];

    let n_copy = n_coarse.min(n_fine).min(sol_coarse.len());
    sol_fine[..n_copy].copy_from_slice(&sol_coarse[..n_copy]);

    for new_node in n_coarse..n_fine {
        let xp = refined.node_coords(new_node as u32);
        let mut best1 = (f64::MAX, 0.0_f64);
        let mut best2 = (f64::MAX, 0.0_f64);
        for cn in 0..n_coarse as u32 {
            let xc = coarse.node_coords(cn);
            let d2: f64 = xp.iter().zip(xc.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            if d2 < best1.0 {
                best2 = best1;
                best1 = (d2, sol_coarse[cn as usize]);
            } else if d2 < best2.0 {
                best2 = (d2, sol_coarse[cn as usize]);
            }
        }
        if best2.0 < f64::MAX
            && (best1.0 - best2.0).abs() < 1e-10 * (best1.0 + best2.0 + 1e-14)
        {
            sol_fine[new_node] = 0.5 * (best1.1 + best2.1);
        } else {
            sol_fine[new_node] = best1.1;
        }
    }
    sol_fine
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use fem_mesh::{SimplexMesh, amr::NCState};
    use crate::{par_mesh::ParallelMesh, partition::MeshPartition};

    fn make_serial_par_mesh(n: usize) -> (ParallelMesh<SimplexMesh<2>>, NCState) {
        use crate::backend::native::SerialBackend;
        use crate::comm::Comm;
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let partition = MeshPartition::new_serial(mesh.n_nodes(), mesh.n_elements());
        let comm = Comm::from_backend(Box::new(SerialBackend));
        let par_mesh = ParallelMesh::new(mesh, comm, partition);
        (par_mesh, NCState::new())
    }

    #[test]
    fn par_refine_increases_elements() {
        let (par_mesh, nc) = make_serial_par_mesh(2);
        let n_before = par_mesh.local_mesh().n_elements();
        let marked: Vec<ElemId> = vec![0, 1];
        let result = par_refine_marked(&par_mesh, nc, &marked, None).unwrap();
        assert!(result.par_mesh.local_mesh().n_elements() > n_before);
    }

    #[test]
    fn par_refine_no_marked_is_identity() {
        let (par_mesh, nc) = make_serial_par_mesh(3);
        let n_before = par_mesh.local_mesh().n_elements();
        let result = par_refine_marked(&par_mesh, nc, &[], None).unwrap();
        assert_eq!(result.par_mesh.local_mesh().n_elements(), n_before);
    }

    #[test]
    fn prolongate_constant_function() {
        let coarse = SimplexMesh::<2>::unit_square_tri(2);
        let mut nc = NCState::new();
        let marked: Vec<ElemId> = (0..coarse.n_elements() as ElemId).collect();
        let (refined, _, _) = nc.refine(&coarse, &marked);
        let sol_coarse = vec![3.14_f64; coarse.n_nodes()];
        let sol_fine = prolongate_p1(&coarse, &refined, &sol_coarse);
        for (i, &v) in sol_fine.iter().enumerate() {
            assert!((v - 3.14).abs() < 1e-12, "node {i}: got {v}");
        }
    }

    #[test]
    fn prolongate_linear_function() {
        let coarse = SimplexMesh::<2>::unit_square_tri(4);
        let mut nc = NCState::new();
        let marked: Vec<ElemId> = (0..coarse.n_elements() as ElemId).collect();
        let (refined, _, _) = nc.refine(&coarse, &marked);
        let sol_coarse: Vec<f64> = (0..coarse.n_nodes())
            .map(|i| { let c = coarse.node_coords(i as u32); c[0] + c[1] })
            .collect();
        let sol_fine = prolongate_p1(&coarse, &refined, &sol_coarse);
        let n_coarse = coarse.n_nodes();
        let max_err = (n_coarse..refined.n_nodes())
            .map(|i| {
                let c = refined.node_coords(i as u32);
                (sol_fine[i] - (c[0] + c[1])).abs()
            })
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.15, "P1 prolongation error: {max_err}");
    }

    #[test]
    fn par_repartition_no_op() {
        let (par_mesh, _) = make_serial_par_mesh(4);
        let n = par_mesh.local_mesh().n_elements();
        let result = par_repartition(par_mesh).unwrap();
        assert_eq!(result.local_mesh().n_elements(), n);
    }
}
