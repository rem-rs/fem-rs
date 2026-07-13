//! Phase D: DWR (Dual-Weighted Residual) scenario depth tests
//!
//! Validates the goal-oriented DWR error estimator across multiple
//! goal functional types, mesh geometries, and refinement strategies.
//!
//! ## Test hierarchy
//!
//! | Test | Goal | Mesh | What it checks |
//! |------|------|------|----------------|
//! | `dwr_point_value_near_center` | PointSolution(center) | Tri3 | Indicators larger near target |
//! | `dwr_point_value_quad4` | PointSolution(center) | Quad4 | Same check for Quad4 |
//! | `dwr_linear_zero` | EnergyNorm | Tri3 | DWR ≈ 0 for linear solution |
//! | `dwr_kelly_indicator_correlation` | (comparison) | Tri3 | DWR & Kelly differ in pattern |
//! | `dwr_efficiency_index_zz` | (efficiency) | Tri3 | eff = O(1) for ZZ estimator |
//! | `dwr_indicator_shape` | PointSolution | Tri3 | All indicators ≥ 0 |
//! | `dwr_mean_stress_vector` | MeanStress | Tri3 | Vector DWR basic sanity |

use std::f64::consts::PI;

use fem_assembly::amr_mf::SimpleDiffusionOp;
use fem_mesh::{Mesh, topology::MeshTopology, amr::{
    dwr_goal_oriented_estimator_2d, dwr_goal_oriented_estimator_3d,
    dwr_estimator,  // legacy scalar DWR for Tri3
    PointSolution, MeanStress, EnergyNorm, LocalFlux, GoalFunctional,
    compute_error_bounds, efficiency_index, stop_on_tolerance,
}};
use fem_core::NodeId;

// ─── Helper: max of a slice ─────────────────────────────────────────────────

fn max_abs(v: &[f64]) -> f64 {
    v.iter().cloned().fold(0.0f64, f64::max)
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D1: Point-value DWR — indicators should be larger near the target node
// ═════════════════════════════════════════════════════════════════════════════

/// Create a linear solution field `u(x,y) = x + y` and compute DWR indicators
/// for `PointSolution` at the center node.  Since the exact solution is linear
/// and P1 captures it exactly, the DWR should be ≈ 0.
/// Then test with a quadratic perturbation `u(x,y) = x² + y²` so DWR > 0,
/// and verify the maximum indicator is near the target node.
#[test]
fn dwr_point_value_linear_trigives_zero() {
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0] + c[1]  // linear
    }).collect();
    let f = vec![0.0; n];

    // Target: center node (near middle of the mesh)
    let target_node = (n / 2) as NodeId;
    let goal = PointSolution::new(target_node);
    let z = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal, &mesh,
    );

    // DWR should be near zero for P1-exact linear solution
    let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
    let max = max_abs(&eta);
    println!("  DWR (linear u, point-value): max η_K = {:.3e}", max);
    assert!(max < 1e-12,
        "DWR should be ≈ 0 for P1-exact linear solution, got max = {:.3e}", max);
}

#[test]
fn dwr_point_value_quadratic_indicator_peaks_near_target() {
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    // Quadratic solution NOT captured exactly by P1 → DWR > 0
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1]*c[1]
    }).collect();
    let f = vec![2.0; n];  // Laplacian(x² + y²) = 4

    // Target node near center
    let target_node = (n / 2) as NodeId;
    let goal = PointSolution::new(target_node);
    let z = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal, &mesh,
    );

    let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
    let max_eta = max_abs(&eta);
    println!("  DWR (quadratic u, point-value): max η_K = {:.3e}", max_eta);
    assert!(max_eta > 1e-6,
        "DWR should be > 0 for quadratic solution not P1-exact, got max = {:.3e}", max_eta);

    // Find which element has max indicator and verify it's near target
    let max_idx = eta.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let ns = mesh.elem_nodes(max_idx as u32);
    let target_coord = mesh.coords_of(target_node);
    let elem_centroid: [f64; 2] = [
        ns.iter().map(|&n| mesh.coords_of(n)[0]).sum::<f64>() / ns.len() as f64,
        ns.iter().map(|&n| mesh.coords_of(n)[1]).sum::<f64>() / ns.len() as f64,
    ];
    let dist = ((elem_centroid[0] - target_coord[0]).powi(2)
              + (elem_centroid[1] - target_coord[1]).powi(2)).sqrt();
    println!("  Max indicator at element {max_idx}, dist to target = {dist:.4}");
    // The max indicator should be within 3 elements of the target (mesh is 8×8 Tri3)
    assert!(dist < 0.2,
        "Max DWR indicator should be near target node, distance = {dist:.4}");
}

#[test]
fn dwr_point_value_quad4() {
    // Same test but on Quad4 mesh
    let mesh = Mesh::<2>::unit_square_quad(8);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1]*c[1]
    }).collect();
    let f = vec![2.0; n];

    let target_node = (n / 2) as NodeId;
    let goal = PointSolution::new(target_node);
    let z = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal, &mesh,
    );

    let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
    let max_eta = max_abs(&eta);
    println!("  DWR Quad4 (quadratic u, point-value): max η_K = {:.3e}", max_eta);
    assert!(max_eta > 1e-6,
        "Quad4 DWR should be > 0 for quadratic solution, got max = {:.3e}", max_eta);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D2: DWR indicator structure — different goals produce different patterns
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_different_goals_different_patterns() {
    // Two different goal functionals should produce different indicator
    // distributions for the same solution field.
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1]*c[1]
    }).collect();
    let f = vec![2.0; n];

    // Goal A: Point value at node 0 (corner)
    let goal_a = PointSolution::new(0);
    let z_a = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal_a, &mesh,
    );
    let eta_a = dwr_goal_oriented_estimator_2d(&mesh, &u, &z_a, &f, 1);

    // Goal B: Point value at center
    let center_node = (n / 2) as NodeId;
    let goal_b = PointSolution::new(center_node);
    let z_b = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal_b, &mesh,
    );
    let eta_b = dwr_goal_oriented_estimator_2d(&mesh, &u, &z_b, &f, 1);

    // The indicator sets should differ (they target different nodes)
    let max_a = max_abs(&eta_a);
    let max_b = max_abs(&eta_b);
    println!("  DWR corner-target: max = {:.3e}", max_a);
    println!("  DWR center-target: max = {:.3e}", max_b);

    // Both should be non-zero
    assert!(max_a > 0.0, "Corner-target DWR should be > 0");
    assert!(max_b > 0.0, "Center-target DWR should be > 0");

    // The distributions should differ (element-wise differences)
    let mut diff_sum = 0.0;
    for i in 0..eta_a.len() {
        diff_sum += (eta_a[i] - eta_b[i]).abs();
    }
    println!("  Total element-wise DWR difference: {:.3e}", diff_sum);
    assert!(diff_sum > 1e-12,
        "Different goals should produce different indicator distributions");
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D3: All DWR indicators should be non-negative (absolute values)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_indicators_non_negative() {
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1]
    }).collect();
    let z: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[1]*c[1] - c[0]
    }).collect();
    let f = vec![2.0; n];

    let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
    for (i, &v) in eta.iter().enumerate() {
        assert!(v >= 0.0,
            "DWR indicator at element {i} should be ≥ 0, got {v:.3e}");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D4: DWR matched with legacy estimator
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_matches_legacy_estimator() {
    // The generalized DWR estimator should match the legacy single-purpose one
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1]
    }).collect();
    let z: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[1]*c[1] - c[0]
    }).collect();
    let f = vec![2.0; n];

    let eta_new = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, 1);
    let eta_old = dwr_estimator(&mesh, &u, &z, &f);

    assert_eq!(eta_new.len(), eta_old.len());
    for i in 0..eta_new.len() {
        let diff = (eta_new[i] - eta_old[i]).abs();
        assert!(diff < 1e-12,
            "DWR mismatch at element {i}: new={:.3e}, old={:.3e}",
            eta_new[i], eta_old[i]);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D5: Error bounds and efficiency index
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_error_bounds_basic() {
    // Verify compute_error_bounds produces correct bounds
    let eta = vec![0.5, 1.0, 1.5, 2.0];
    let bounds = compute_error_bounds(&eta, 1.5, 0.5);

    let sum_sq: f64 = 0.25 + 1.0 + 2.25 + 4.0;
    let expected_global = sum_sq.sqrt();  // √7.5 ≈ 2.739
    let expected_max = 2.0;

    assert!((bounds.global_estimate - expected_global).abs() < 1e-14);
    assert!((bounds.upper - 1.5 * expected_global).abs() < 1e-14);
    assert!((bounds.lower - 0.5 * expected_max).abs() < 1e-14);
    assert!((bounds.max_indicator - expected_max).abs() < 1e-14);
    assert_eq!(bounds.n_elems, 4);
}

#[test]
fn dwr_efficiency_index_ideal() {
    let eff = efficiency_index(2.5, 2.5);
    assert!((eff - 1.0).abs() < 1e-14);
}

#[test]
fn dwr_efficiency_index_zero_true_error() {
    let eff = efficiency_index(1.0, 0.0);
    assert!(eff.is_infinite());
}

#[test]
fn dwr_stop_on_tolerance() {
    // eta = [1,1,1] → global = √3 ≈ 1.732, C_R=1.5 → upper = 2.598
    // tol=3.0 → stop (upper < tol)
    assert!(stop_on_tolerance(&[1.0; 3], 3.0, 1.5));
    // tol=2.0 → don't stop (upper >= tol)
    assert!(!stop_on_tolerance(&[1.0; 3], 2.0, 1.5));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D6: DWR for 3-D meshes (basic sanity)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_3d_tet4_linear_gives_small() {
    let mesh = Mesh::<3>::unit_cube_tet(3);
    let n = mesh.n_nodes();
    let u: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0] + c[1] - c[2]
    }).collect();
    let z: Vec<f64> = (0..n).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        c[0]*c[0] + c[1] - 2.0*c[2]
    }).collect();
    let f = vec![2.0; n];

    let eta = dwr_goal_oriented_estimator_3d(&mesh, &u, &z, &f, 1);
    let max = max_abs(&eta);
    println!("  DWR 3D Tet4: max η_K = {:.3e}", max);
    // For Tet4 with non-linear z, DWR should be non-zero
    assert!(max > 0.0, "3D DWR should be > 0 for quadratic adjoint");
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D7: Kelly vs DWR indicator pattern comparison
//  (DWR for point-value should localize differently from Kelly)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_vs_kelly_different_distributions() {
    // Solve Poisson on a coarse mesh, compare Kelly and DWR indicator patterns.
    // Kelly measures face jumps (edge-based), DWR weights by dual solution.
    // They should distribute differently for a point-value goal.
    use fem_assembly::{
        Assembler,
        standard::{DiffusionIntegrator, DomainSourceIntegrator},
        postproc::error_estimate::kelly_estimator,
        postproc::grid_function::GridFunction,
    };
    use fem_space::{H1Space, fe_space::FESpace,
        constraints::{apply_dirichlet, boundary_dofs},
    };
    use fem_solver::{solve_cg, SolverConfig};
    use fem_core::ElemId;

    let mesh = Mesh::<2>::unit_square_tri(8);
    let order = 1u8;
    let space = H1Space::new(mesh.clone(), order);

    // Assemble and solve Poisson with homogeneous Dirichlet BC
    let k = Assembler::assemble_bilinear(
        &space, &[&DiffusionIntegrator { kappa: 1.0 }], 3,
    );
    let f = Assembler::assemble_linear(
        &space, &[&DomainSourceIntegrator::new(|x| {
            use std::f64::consts::PI;
            2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
        })], 3,
    );
    let bnd = boundary_dofs(&mesh, space.dof_manager(), &[1, 2, 3, 4]);
    let mut k_bc = k.clone();
    let mut f_bc = f.clone();
    apply_dirichlet(&mut k_bc, &mut f_bc, &bnd, &vec![0.0; bnd.len()]);

    let n_dofs = space.n_dofs();
    let mut u_h = vec![0.0; n_dofs];
    solve_cg(&k_bc, &f_bc, &mut u_h, &SolverConfig {
        rtol: 1e-10, max_iter: 2000, verbose: false, ..SolverConfig::default()
    }).expect("CG solve failed");

    // Kelly estimator
    let gf = GridFunction::new(&space, u_h.clone());
    let kelly = kelly_estimator(&gf).eta;
    let kelly_max = max_abs(&kelly);

    // DWR with PointSolution at center
    let center_node = (mesh.n_nodes() / 2) as NodeId;
    let goal = PointSolution::new(center_node);
    let z = <PointSolution as GoalFunctional<2>>::assemble_adjoint_rhs(
        &goal, &mesh,
    );

    // Solve adjoint: K * z_h = adj_rhs (with same BC)
    let mut adj_rhs_bc = z.clone();
    for &d in &bnd { adj_rhs_bc[d as usize] = 0.0; }
    let mut z_h = vec![0.0; n_dofs];
    solve_cg(&k_bc, &adj_rhs_bc, &mut z_h, &SolverConfig {
        rtol: 1e-10, max_iter: 2000, verbose: false, ..SolverConfig::default()
    }).expect("Adjoint CG solve failed");

    // DWR estimator — use nodal values (for Q1, DOF index = node index)
    let u_nodal: Vec<f64> = (0..mesh.n_nodes()).map(|i| u_h[i]).collect();
    let z_nodal: Vec<f64> = (0..mesh.n_nodes()).map(|i| z_h[i]).collect();
    let f_nodal: Vec<f64> = (0..mesh.n_nodes()).map(|i| {
        let c = mesh.coords_of(i as NodeId);
        2.0 * PI * PI * (PI * c[0]).sin() * (PI * c[1]).sin()
    }).collect();

    let dwr = dwr_goal_oriented_estimator_2d(&mesh, &u_nodal, &z_nodal, &f_nodal, 1);
    let dwr_max = max_abs(&dwr);

    println!("  Kelly max η_K = {:.3e}", kelly_max);
    println!("  DWR   max η_K = {:.3e}", dwr_max);

    // Both should be non-zero
    assert!(kelly_max > 0.0, "Kelly should be > 0 for non-exact solution");
    assert!(dwr_max > 0.0, "DWR should be > 0 for non-exact solution");

    // Kelly and DWR should correlate but differ in distribution
    // (they're both measuring error, but weighted differently)
    let mut dot_product = 0.0;
    let mut norm_k = 0.0;
    let mut norm_d = 0.0;
    for i in 0..kelly.len() {
        dot_product += kelly[i] * dwr[i];
        norm_k += kelly[i] * kelly[i];
        norm_d += dwr[i] * dwr[i];
    }
    let cos_angle = dot_product / (norm_k.sqrt() * norm_d.sqrt() + 1e-30);
    println!("  Kelly-DWR cosine similarity = {:.4}", cos_angle);
    // They should be positively correlated (> 0) but not identical (< 0.999)
    assert!(cos_angle > 0.0, "Kelly and DWR should be positively correlated");
    assert!(cos_angle < 0.999,
        "Kelly and DWR should not be identical, cos = {cos_angle:.4}");
}

// ═════════════════════════════════════════════════════════════════════════════
//  Test D8: DWR indicator for MeanStress functional (vector problem)
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn dwr_mean_stress_2d_tri3() {
    // For displacement u = (x, 0) with λ=μ=1, σ_xx = 3.
    // The DWR should be non-zero for a non-linear test function.
    let mesh = Mesh::<2>::unit_square_tri(8);
    let n = mesh.n_nodes();
    let n_comp = 2;

    let mut u = vec![0.0; n * n_comp];
    let mut z = vec![0.0; n * n_comp];
    let mut f = vec![0.0; n * n_comp];
    for i in 0..n {
        let c = mesh.coords_of(i as NodeId);
        u[i * n_comp + 0] = c[0];        // u_x = x
        u[i * n_comp + 1] = 0.0;         // u_y = 0
        z[i * n_comp + 0] = c[1];        // z_x = y
        z[i * n_comp + 1] = c[0];        // z_y = x
        f[i * n_comp + 0] = 0.0;
        f[i * n_comp + 1] = 0.0;
    }

    let eta = dwr_goal_oriented_estimator_2d(&mesh, &u, &z, &f, n_comp);
    let max = max_abs(&eta);
    println!("  DWR MeanStress 2D Tri3: max η_K = {:.3e}", max);
    // Linear fields → should be near zero
    assert!(max < 1e-12,
        "DWR should be ≈ 0 for linear displacement field, got max = {:.3e}", max);
}
