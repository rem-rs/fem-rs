//! Tests for the [`Hybridization`] port (MFEM `fem/hybridization.cpp`).
//!
//! The acceptance check: on the H(div) diffusion problem
//! `-∇(∇·F) + F = f` with `F·n = 0` (MFEM ex4 setting), the hybridized
//! solve must reproduce the direct assembled solve — both operate on the
//! same discrete operator, so the dof vectors must agree to solver
//! tolerance.

use nalgebra::{DMatrix, DVector};
use fem_linalg::CsrMatrix;
use fem_mesh::{Mesh, MeshTopology};
use fem_space::constraints::{boundary_dofs_hdiv, form_linear_system};
use fem_space::fe_space::FESpace;
use fem_space::HDivSpace;

use crate::hybridization::{ConstraintIntegratorKind, Hybridization, TraceSpaceKind};
use crate::standard::{GradDivIntegrator, VectorMassIntegrator};
use crate::vector_assembler::VectorAssembler;
use crate::vector_integrator::{VectorLinearIntegrator, VectorQpData};

use super::vector_element_matrix;

// ─── Manufactured source: f = −(π + 2π³)(sin πx cos πy, cos πx sin πy) ────
//
// Exact solution F = −π(sin πx cos πy, cos πx sin πy) satisfies F·n = 0 on
// the unit square and −∇(∇·F) + F = f.

struct ManufacturedSource;

impl VectorLinearIntegrator for ManufacturedSource {
    fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f: &mut [f64]) {
        let x = qp.x_phys;
        let c = std::f64::consts::PI + 2.0 * std::f64::consts::PI.powi(3);
        let fx = -c * (std::f64::consts::PI * x[0]).sin() * (std::f64::consts::PI * x[1]).cos();
        let fy = -c * (std::f64::consts::PI * x[1]).sin() * (std::f64::consts::PI * x[0]).cos();
        for i in 0..qp.n_dofs {
            f[i] += qp.weight * (qp.phi_vec[i * 2] * fx + qp.phi_vec[i * 2 + 1] * fy);
        }
    }
}

/// Direct solve of the H(div) system with `F·n = 0` essential conditions
/// (dense LU, the meshes are tiny).
fn solve_direct(
    space: &HDivSpace<Mesh<2>>,
    ess: &[u32],
    a: &CsrMatrix<f64>,
    rhs: &[f64],
) -> Vec<f64> {
    let mut mat = a.clone();
    let mut f = rhs.to_vec();
    let mut x = vec![0.0_f64; space.n_dofs()];
    let zeros = vec![0.0_f64; ess.len()];
    form_linear_system(&mut mat, &mut f, &mut x, ess, &zeros);
    let n = mat.nrows;
    let mut dm = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for p in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            dm[(i, mat.col_idx[p] as usize)] = mat.values[p];
        }
    }
    let sol = dm.lu().solve(&DVector::from_column_slice(&f)).unwrap();
    sol.as_slice().to_vec()
}

/// Assemble the H(div) diffusion operator and RHS for the test problem.
fn setup(space: &HDivSpace<Mesh<2>>) -> (CsrMatrix<f64>, Vec<f64>) {
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    let mat = VectorAssembler::assemble_bilinear(space, &[&grad_div, &vec_mass], 2);
    let rhs = VectorAssembler::assemble_linear(space, &[&ManufacturedSource], 2);
    (mat, rhs)
}

fn ess_boundary_dofs(space: &HDivSpace<Mesh<2>>) -> Vec<u32> {
    let tags: Vec<i32> = space.mesh().unique_boundary_tags();
    boundary_dofs_hdiv(space.mesh(), space, &tags)
}

/// Solve the H(div) problem through the hybridization and compare against
/// the direct assembled solve (MFEM ex4 hybridization route, RT0).
#[test]
fn hdiv_hybrid_matches_direct_face_dg_trace() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = HDivSpace::new(mesh, 0); // RT0
    let ess = ess_boundary_dofs(&space);
    let (mat, rhs) = setup(&space);

    let x_direct = solve_direct(&space, &ess, &mat, &rhs);

    // Hybridization: per-face P0 trace (MFEM DG_Interface(order-1), ex4).
    let mut hyb = Hybridization::new(
        TraceSpaceKind::FaceDG { order: 0 },
        ConstraintIntegratorKind::NormalTraceJump,
    );
    hyb.init(&space, &ess);
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();
    let h = hyb.get_matrix().expect("hybridized matrix after finalize").clone();

    let b_r = hyb.reduce_rhs(&rhs);
    let lam = {
        let mut dm = DMatrix::<f64>::zeros(h.nrows, h.ncols);
        for i in 0..h.nrows {
            for p in h.row_ptr[i]..h.row_ptr[i + 1] {
                dm[(i, h.col_idx[p] as usize)] = h.values[p];
            }
        }
        dm.lu().solve(&DVector::from_column_slice(&b_r)).unwrap()
    };
    let mut x_hyb = vec![0.0_f64; space.n_dofs()];
    hyb.compute_solution(&rhs, lam.as_slice(), &mut x_hyb);

    let max_diff = x_direct
        .iter()
        .zip(x_hyb.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-10,
        "hybridized solution must match the direct solve (max diff {max_diff:.3e})"
    );
}

/// Same with a P1-per-face DG trace (`FaceDG { order: 1 }`): exercises the
/// multi-dof-per-face path (linear trace shape functions along the face).
/// (The `H1` trace pairing with RT0 is rank-deficient — each interior face
/// contributes only one independent constraint because the RT0 flux is
/// constant and the two vertex dofs share it — so `H` is singular there
/// and a direct-match test would be ill-posed; that degeneracy is inherent
/// to the c_fes choice, not to the port.)
#[test]
fn hdiv_hybrid_matches_direct_face_dg_p1_trace() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = HDivSpace::new(mesh, 0);
    let ess = ess_boundary_dofs(&space);
    let (mat, rhs) = setup(&space);

    let x_direct = solve_direct(&space, &ess, &mat, &rhs);

    let mut hyb = Hybridization::new(
        TraceSpaceKind::FaceDG { order: 1 },
        ConstraintIntegratorKind::NormalTraceJump,
    );
    hyb.init(&space, &ess);
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();
    let h = hyb.get_matrix().unwrap().clone();
    assert_eq!(h.nrows, 16, "two trace dofs per interior face");

    let b_r = hyb.reduce_rhs(&rhs);
    // The P1 trace has two multipliers per face while the RT0 flux on the
    // face carries only one moment: H is rank 16 (consistent, redundant);
    // solve in the min-norm sense.
    let lam = {
        let mut dm = DMatrix::<f64>::zeros(h.nrows, h.ncols);
        for i in 0..h.nrows {
            for p in h.row_ptr[i]..h.row_ptr[i + 1] {
                dm[(i, h.col_idx[p] as usize)] = h.values[p];
            }
        }
        let sv = dm.clone().svd(true, true);
        sv.solve(&DVector::from_column_slice(&b_r), 1e-10).unwrap()
    };
    let mut x_hyb = vec![0.0_f64; space.n_dofs()];
    hyb.compute_solution(&rhs, lam.as_slice(), &mut x_hyb);

    let max_diff = x_direct
        .iter()
        .zip(x_hyb.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-10,
        "P1-trace hybridized solution must match the direct solve (max diff {max_diff:.3e})"
    );
}

/// Constraint-matrix values: for RT0 + P0 trace, the C block of an interior
/// face is `[+1; −1]` (the two RT0 face dofs carry unit outward flux and
/// the MFEM CalcOrtho length cancels the 1/|e| of the RT0 basis).
#[test]
fn ct_values_rt0_p0_unit_flux() {
    let mesh = Mesh::<2>::unit_square_tri(1); // 2 triangles, 1 interior edge
    let space = HDivSpace::new(mesh, 0);
    let ess = ess_boundary_dofs(&space);

    let mut hyb = Hybridization::new(
        TraceSpaceKind::FaceDG { order: 0 },
        ConstraintIntegratorKind::NormalTraceJump,
    );
    hyb.init(&space, &ess);
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();
    let h = hyb.get_matrix().unwrap();
    assert_eq!(h.nrows, 1, "one interior face → one P0 trace dof");
    assert!(h.values[0] > 0.0, "H must be positive (Cb S⁻¹ Cbᵀ)");

    // Each element's free block is the 1×1 block of its interior-edge RT0
    // dof (the boundary-edge dofs are essential); C = ±1 per element, so
    // H = Σ_e 1/A[l_e, l_e] with l_e the interior-edge local index.
    let ess_set: std::collections::HashSet<u32> = ess.iter().copied().collect();
    let mut expected = 0.0;
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        let dofs = space.element_dofs(e);
        let l = (0..dofs.len())
            .find(|&j| !ess_set.contains(&dofs[j]))
            .expect("RT0 element must keep its interior-edge dof");
        expected += 1.0 / elmat[l * dofs.len() + l];
    }
    assert!(
        (h.values[0] - expected).abs() < 1e-12 * expected.abs().max(1.0),
        "H = Σ 1/A_e mismatch: {} vs {expected}",
        h.values[0]
    );
}

/// Essential dofs are excluded from the free blocks: with all boundary RT0
/// edge dofs essential, the hat dofs sitting on boundary edges must be
/// marked essential and never appear in the hybridized matrix rows.
#[test]
fn essential_hat_dofs_are_excluded() {
    let mesh = Mesh::<2>::unit_square_tri(2);
    let space = HDivSpace::new(mesh, 0);
    let ess = ess_boundary_dofs(&space);

    let mut hyb = Hybridization::new(
        TraceSpaceKind::FaceDG { order: 0 },
        ConstraintIntegratorKind::NormalTraceJump,
    );
    hyb.init(&space, &ess);
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();

    // RT0: every essential dof is a boundary-edge face dof; the interior
    // RT0 face dofs (shared by two elements) are the C rows.  Euler for
    // the 8-triangle mesh: 16 edges, 8 on the boundary → 8 interior.
    assert_eq!(hyb.n_trace_dofs(), 8, "unit_square_tri(2) has 8 interior edges");
    assert_eq!(hyb.get_matrix().unwrap().nrows, 8);

    // Quadrature-order sanity: H must be SPD → CG converges quickly.
    let h = hyb.get_matrix().unwrap();
    for i in 0..h.nrows {
        let mut diag = 0.0;
        for p in h.row_ptr[i]..h.row_ptr[i + 1] {
            if h.col_idx[p] as usize == i {
                diag = h.values[p];
            }
        }
        assert!(diag > 0.0, "H diagonal must be positive (row {i})");
    }
}

/// Reset clears the hybridized matrix but keeps the constraint data
/// (MFEM `Reset`), and re-assembly through `finalize` restores it.
#[test]
fn reset_and_reassemble() {
    let mesh = Mesh::<2>::unit_square_tri(1);
    let space = HDivSpace::new(mesh, 0);
    let ess = ess_boundary_dofs(&space);

    let mut hyb = Hybridization::new(
        TraceSpaceKind::FaceDG { order: 0 },
        ConstraintIntegratorKind::NormalTraceJump,
    );
    hyb.init(&space, &ess);
    let grad_div = GradDivIntegrator { kappa: 1.0 };
    let vec_mass = VectorMassIntegrator { alpha: 1.0 };
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();
    assert!(hyb.get_matrix().is_some());

    hyb.reset();
    assert!(hyb.get_matrix().is_none());

    // Re-assemble and finalize again (init is a no-op now).
    for e in 0..space.mesh().n_elements() as u32 {
        let elmat = vector_element_matrix(&space, e, &[&grad_div, &vec_mass], 2);
        hyb.assemble_matrix(e as usize, &elmat);
    }
    hyb.finalize();
    assert!(hyb.get_matrix().is_some());
}
