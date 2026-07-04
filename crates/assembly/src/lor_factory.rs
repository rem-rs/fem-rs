//! Convenient factory for building LOR-AMG preconditioners from high-order H¹ spaces.
//!
//! Bridges the gap between `fem-space` (H1Space), `fem-assembly` (prolongation),
//! and `fem-solver` (LorAmgPrecond).  Users can go from a Pk H1Space directly to
//! a working LOR-AMG solver without manually building the prolongation matrix P.
//!
//! # Usage
//! ```rust,ignore
//! use fem_assembly::lor_factory::build_lor_amg_h1;
//!
//! let pk = H1Space::new(mesh, 3);              // P3 space
//! let a_ho = assembler.assemble(&pk);          // high-order stiffness matrix
//! let lor = build_lor_amg_h1(&pk, &a_ho, None).unwrap();
//! let cfg = SolverConfig { rtol: 1e-8, max_iter: 50, ..Default::default() };
//! let res = solve_pcg_lor_amg(&a_ho, &b, &mut x, &lor, &cfg).unwrap();
//! ```

use fem_core::FemResult;
use fem_linalg::CsrMatrix;
use fem_mesh::SimplexMesh;
use fem_solver::lor::{AmgConfig, LorAmgPrecond};
use fem_space::fe_space::FESpace;
use fem_space::h1::H1Space;

use crate::transfer::build_prolongation_h1;

/// Build a LOR-AMG preconditioner for a 2-D high-order H¹ space.
///
/// This function:
/// 1. Creates a P1 H1Space on the same mesh (the "low-order refined" space).
/// 2. Builds the prolongation `P: P1 → Pk` via `build_prolongation_h1`.
/// 3. Builds the LOR-AMG preconditioner `M⁻¹ = P · A_LO⁻¹ · Pᵀ`
///    where `A_LO = Pᵀ · A_HO · P`.
///
/// # Arguments
/// * `pk_space` — high-order H¹ space (e.g. P2, P3, …).
/// * `a_ho`    — assembled high-order system matrix (SPD).
/// * `amg_cfg` — optional AMG configuration.  Uses `AmgConfig::default()` when `None`.
///
/// # Returns
/// `Ok(LorAmgPrecond)` or `Err` if the prolongation could not be built.
pub fn build_lor_amg_h1(
    pk_space: &H1Space<SimplexMesh<2>>,
    a_ho: &CsrMatrix<f64>,
    amg_cfg: Option<AmgConfig>,
) -> FemResult<LorAmgPrecond> {
    if pk_space.order() <= 1 {
        return Err(fem_core::FemError::Other(
            "build_lor_amg_h1: space order must be >= 2 (P1 needs no LOR)".into(),
        ));
    }

    // Build the low-order P1 space on the same mesh.
    let p1 = H1Space::new(pk_space.mesh().clone(), 1);

    // Build prolongation P: P1 → Pk.
    let tol = 0.1;  // point-location tolerance on reference element
    let (p, stats) = build_prolongation_h1(&p1, pk_space, tol);

    if stats.located_count == 0 {
        return Err(fem_core::FemError::Other(
            "build_lor_amg_h1: prolongation located 0 DOFs — mesh mismatch?".into(),
        ));
    }

    let amg_cfg = amg_cfg.unwrap_or_default();

    Ok(LorAmgPrecond::build(a_ho, &p, &amg_cfg))
}

/// Build a LOR-AMG preconditioner for a 3-D high-order H¹ space (Tet4 mesh).
pub fn build_lor_amg_h1_3d(
    pk_space: &H1Space<SimplexMesh<3>>,
    a_ho: &CsrMatrix<f64>,
    amg_cfg: Option<AmgConfig>,
) -> FemResult<LorAmgPrecond> {
    if pk_space.order() <= 1 {
        return Err(fem_core::FemError::Other(
            "build_lor_amg_h1_3d: space order must be >= 2".into(),
        ));
    }

    let p1 = H1Space::new(pk_space.mesh().clone(), 1);

    use crate::transfer::build_prolongation_h1_3d;
    let tol = 0.1;
    let (p, stats) = build_prolongation_h1_3d(&p1, pk_space, tol);

    if stats.located_count == 0 {
        return Err(fem_core::FemError::Other(
            "build_lor_amg_h1_3d: prolongation located 0 DOFs".into(),
        ));
    }

    let amg_cfg = amg_cfg.unwrap_or_default();

    Ok(LorAmgPrecond::build(a_ho, &p, &amg_cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_lor_amg_h1_p2_smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let pk = H1Space::new(mesh, 2);
        let na = pk.n_dofs();
        // Build a simple SPD matrix (identity for smoke test)
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0_f64 + (i as f64 % 5.0) * 0.2); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1 failed: {:?}", lor.err());
    }

    #[test]
    fn build_lor_amg_h1_p3_smoke() {
        let mesh = SimplexMesh::<2>::unit_square_tri(4);
        let pk = H1Space::new(mesh, 3);
        let na = pk.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1 p=3 failed: {:?}", lor.err());
    }

    #[test]
    fn build_lor_amg_h1_3d_p2_smoke() {
        let mesh = SimplexMesh::<3>::unit_cube_tet(3);
        let pk = H1Space::new(mesh, 2);
        let na = pk.n_dofs();
        let mut coo = fem_linalg::CooMatrix::<f64>::new(na, na);
        for i in 0..na { coo.add(i, i, 1.0); }
        let a_ho = coo.into_csr();
        let lor = build_lor_amg_h1_3d(&pk, &a_ho, None);
        assert!(lor.is_ok(), "build_lor_amg_h1_3d p=2 failed: {:?}", lor.err());
    }
}
