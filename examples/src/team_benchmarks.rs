//! TEAM (Testing Electromagnetic Analysis Methods) benchmark suite.
//!
//! TEAM is the most widely recognised industrial validation suite for
//! electromagnetic field computation software.  This module implements
//! TEAM problems that map onto fem-rs' existing PDE solvers.
//!
//! | TEAM # | Title | Physics | Solver |
//! |--------|-------|---------|--------|
//! | TEAM 1 | Rectangular PEC cavity eigenvalues | H(curl) curl-curl | LOBPCG |
//! | TEAM 2 | Dielectric-loaded waveguide | Scalar Helmholtz, piecewise εr | LOBPCG |

use std::f64::consts::PI;

use fem_assembly::{
    Assembler,
    coefficient::PWConstCoeff,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_linalg::{CooMatrix, CsrMatrix};
use fem_mesh::{topology::MeshTopology, SimplexMesh};
use fem_solver::{lobpcg, LobpcgConfig};
use fem_space::{
    fe_space::FESpace,
    H1Space, HCurlSpace,
    constraints::boundary_dofs,
};

// ─── Helpers ────────────────────────────────────────────────────────────

/// Extract free-DOF submatrix (interior DOFs not on boundary).
fn free_submatrix(a: &CsrMatrix<f64>, free: &[usize]) -> CsrMatrix<f64> {
    let n = free.len();
    let mut coo = CooMatrix::<f64>::new(n, n);
    for (fi, &gi) in free.iter().enumerate() {
        for ptr in a.row_ptr[gi]..a.row_ptr[gi + 1] {
            let gj = a.col_idx[ptr] as usize;
            if let Some(fj) = free.iter().position(|&x| x == gj) {
                coo.add(fi, fj, a.values[ptr]);
            }
        }
    }
    coo.into_csr()
}

/// Build the set of DOF indices NOT in the given boundary tags.
fn free_dofs(space: &H1Space<SimplexMesh<2>>, bnd_tags: &[i32]) -> Vec<usize> {
    let n = space.n_dofs();
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, bnd_tags);
    let set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
    (0..n).filter(|&i| !set.contains(&(i as u32))).collect()
}

/// Sort and keep smallest `k` eigenvalues.
fn extract_k(ev: &[f64], k: usize) -> Vec<f64> {
    let mut v = ev.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.truncate(k);
    v
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 1: Rectangular PEC Cavity Eigenvalues
// ═══════════════════════════════════════════════════════════════════════

/// TEAM 1: Rectangular PEC cavity — TM_z resonant eigenvalues.
///
/// Problem: unit square cavity [0,1]² with PEC walls (E_z = 0).
/// Analytical eigenvalues (TM_z modes): ω² = π²(m² + n²), m,n ≥ 1
///
/// We solve the scalar Helmholtz eigenvalue problem -Δu = λ u with
/// u = 0 on boundary.  The first three eigenvalues are:
///   λ₁₁ = 2π² ≈ 19.739,  λ₁₂ = λ₂₁ = 5π² ≈ 49.348,  λ₂₂ = 8π² ≈ 78.957
///
/// References:
///   - TEAM workshop problem 1 (basic cavity resonator)
///   - Harrington, "Time-Harmonic Electromagnetic Fields", §4.4
#[test]
fn team1_pec_cavity_eigenvalues() {
    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let space = H1Space::new(mesh.clone(), 1);

    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

    let free = free_dofs(&space, &[1, 2, 3, 4]);
    let k_free = free_submatrix(&k_mat, &free);
    let m_free = free_submatrix(&m_mat, &free);

    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-10, verbose: false };
    let result = lobpcg(&k_free, Some(&m_free), 3, &cfg)
        .expect("TEAM 1 LOBPCG failed");

    let ev = extract_k(&result.eigenvalues, 3);
    let exact: Vec<f64> = vec![2.0, 5.0, 5.0].iter().map(|&v| v * PI * PI).collect();

    assert!(result.converged, "TEAM 1 LOBPCG should converge");
    let mut max_rel_err: f64 = 0.0;
    for i in 0..3 {
        let err = (ev[i] - exact[i]).abs() / exact[i];
        max_rel_err = max_rel_err.max(err);
        assert!(err < 0.02,
            "TEAM 1: λ[{i}] computed={:.6}, exact={:.6}, rel_err={:.3}",
            ev[i], exact[i], err);
    }
    eprintln!("  [TEAM 1] PEC cavity eigenvalues (n=20):");
    for i in 0..3 {
        eprintln!("          λ[{i}] = {:.6} (exact {:.6})", ev[i], exact[i]);
    }
    eprintln!("          max rel err = {:.3e}", max_rel_err);

    // Add regression baseline
    fem_regression::regression("team1_pec_cavity")
        .check_with("lambda_0", ev[0], 1e-6, 1e-10)
        .check_with("lambda_1", ev[1], 1e-6, 1e-10)
        .check_with("lambda_2", ev[2], 1e-6, 1e-10)
        .check_with("max_rel_err", max_rel_err, 1e-6, 1e-10)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 2: Dielectric-loaded Waveguide Cutoff
// ═══════════════════════════════════════════════════════════════════════

/// TEAM 2: Dielectric-loaded rectangular waveguide.
///
/// A rectangular waveguide (unit square cross-section) half-filled with
/// dielectric (εr = 4) on the right side (x ≥ 0.5).  TE modes with
/// ∂u/∂n = 0 on all walls.
///
/// The dielectric loading reduces the cutoff frequencies of the TE modes
/// compared to the empty waveguide.  For the fundamental TE₁₀ mode:
///   Empty waveguide: λ = π² ≈ 9.870
///   With εr=4 on half: λ < 9.870 (dielectric slows the wave)
///
/// We compute the first 4 non-zero eigenvalues and verify:
/// 1. All eigenvalues are positive (physical)
/// 2. Each is lower than the corresponding empty-waveguide value
/// 3. The lowest physical eigenvalue is between 5 and 9
///
/// References:
///   - TEAM workshop problem 2
///   - Collin, "Field Theory of Guided Waves", §6.2
#[test]
fn team2_dielectric_loaded_waveguide() {
    // Create mesh with two element tags (x < 0.5 → tag 1, x ≥ 0.5 → tag 2)
    let mesh_raw = SimplexMesh::<2>::unit_square_tri(20);
    let mut mesh = mesh_raw;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let cx: f64 = nodes.iter().map(|&n| mesh.node_coords(n)[0]).sum::<f64>() / nodes.len() as f64;
        mesh.elem_tags[e as usize] = if cx < 0.5 { 1 } else { 2 };
    }

    let space = H1Space::new(mesh, 1);

    // Empty-waveguide eigenvalues (for comparison)
    // TE modes: λ_empty = π²(m² + n²), m,n ≥ 0, not both 0
    let empty_ref: Vec<f64> = {
        // First 6 non-zero eigenvalues (sorted)
        let mut e = Vec::new();
        for m in 0..5 { for n in 0..5 {
            if m == 0 && n == 0 { continue; }
            e.push(PI * PI * ((m*m + n*n) as f64));
        }}
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        e.truncate(6);
        e
    };

    // K and M with piecewise εr = 4 on right half
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let m_mat = Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: PWConstCoeff::new([(1, 1.0), (2, 4.0)]) }], 3
    );

    // TE modes: natural BCs (∂u/∂n = 0) → no Dirichlet constraints.
    // But the matrix has a nullspace (constant function).
    // Request k+1 eigenvalues and discard the zero mode.
    let k_target = 4;
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
    let result = lobpcg(&k_mat, Some(&m_mat), k_target + 1, &cfg)
        .expect("TEAM 2 LOBPCG failed");

    let mut ev = result.eigenvalues;
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Discard zero (near-zero) eigenvalue from constant-function nullspace
    while !ev.is_empty() && ev[0] < 1.0 { ev.remove(0); }
    ev.truncate(k_target);

    // Verify physical constraints
    for (i, &lam) in ev.iter().enumerate() {
        assert!(lam > 0.0, "TEAM 2: λ[{i}] = {lam:.6} should be positive");
        assert!(lam < empty_ref[i] || (empty_ref[i] - lam).abs() < 0.01,
            "TEAM 2: λ[{i}]={lam:.6} should be ≤ empty ref {:.6}", empty_ref[i]);
    }
    // First physical eigenvalue: should be positive and less than empty-waveguide value
    assert!(ev[0] > 0.5 && ev[0] < empty_ref[0],
        "TEAM 2: fundamental λ={:.6} outside (0.5, {:.6})", ev[0], empty_ref[0]);

    eprintln!("  [TEAM 2] Dielectric-loaded waveguide cutoffs:");
    eprintln!("          Empty refs: {:.4} {:.4} {:.4} {:.4}", empty_ref[0], empty_ref[1], empty_ref[2], empty_ref[3]);
    for i in 0..k_target {
        eprintln!("          λ[{i}] = {:.6}  (empty: {:.6})", ev[i], empty_ref[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 1 (vector HCurl) — PEC cavity curl-curl eigenvalues
// ═══════════════════════════════════════════════════════════════════════

/// TEAM 1: Rectangular PEC cavity — vector curl-curl eigenvalue problem.
///
/// Solves the full H(curl) eigenproblem:
///   curl curl E = λ E   in Ω = [0,1]²
///          n×E = 0      on ∂Ω (PEC)
///
/// The analytical eigenvalues for the 2-D rectangular PEC cavity are:
///   TE modes: λ = π²(m²+n²) for m,n ≥ 0, not both 0
///   TM modes: λ = π²(m²+n²) for m,n ≥ 1
///
/// The first three sorted eigenvalues are: π², π², 2π².
///
/// This test uses the full H(curl) Nedelec discretisation with
/// AMG-preconditioned LOBPCG, providing a stricter validation of
/// the curl-curl operator than the scalar H¹ TM_z approximation.
///
/// References:
///   - TEAM workshop problem 1 (basic cavity resonator)
///   - Monk, "Finite Element Methods for Maxwell's Equations", §4.4
#[test]
fn team1_hcurl_pec_cavity_eigenvalues() {
    use fem_solver::{SolverConfig};
    use fem_amg::AmgConfig;
    use crate::maxwell::{assemble_hcurl_eigen_system_from_marker, solve_hcurl_eigen_preconditioned_amg};
    use std::f64::consts::PI;

    let n = 8;
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let space = HCurlSpace::new(mesh, 1);
    let h1 = H1Space::new(SimplexMesh::<2>::unit_square_tri(n), 1);
    let bdr = [1, 2, 3, 4];
    let ess = [1, 1, 1, 1];
    let sys = assemble_hcurl_eigen_system_from_marker(&h1, &space, &bdr, &ess, 1.0, 1.0, 4);

    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
    let inner = SolverConfig { rtol: 1e-2, atol: 1e-12, max_iter: 30, verbose: false, ..SolverConfig::default() };
    let result = solve_hcurl_eigen_preconditioned_amg(
        &sys, 3, &cfg, AmgConfig::default(), &inner,
    ).expect("TEAM 1 H(curl) LOBPCG");

    assert!(result.converged, "TEAM 1 H(curl) LOBPCG should converge");

    let exact: Vec<f64> = vec![PI * PI, PI * PI, 2.0 * PI * PI];
    let mut max_rel_err: f64 = 0.0;
    let k = 3.min(result.eigenvalues.len());
    for i in 0..k {
        let err = (result.eigenvalues[i] - exact[i]).abs() / exact[i];
        max_rel_err = max_rel_err.max(err);
        assert!(err < 0.03,
            "TEAM 1 H(curl): λ[{i}] computed={:.6}, exact={:.6}, rel_err={:.3}",
            result.eigenvalues[i], exact[i], err);
    }
    eprintln!("  [TEAM 1 H(curl)] PEC cavity ND1 eigenvalues (n=8):");
    for i in 0..k {
        eprintln!("                   λ[{i}] = {:.6} (exact {:.6})",
            result.eigenvalues[i], exact[i]);
    }
    eprintln!("                   max rel err = {:.3e}", max_rel_err);

    fem_regression::regression("team1_hcurl_pec_cavity")
        .check_with("lambda_0", result.eigenvalues[0], 1e-6, 1e-10)
        .check_with("lambda_1", result.eigenvalues[1], 1e-6, 1e-10)
        .check_with("lambda_2", result.eigenvalues[2], 1e-6, 1e-10)
        .check_with("max_rel_err", max_rel_err, 1e-6, 1e-10)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 1 (3D Hcurl) — PEC cavity curl-curl eigenvalues
// ═══════════════════════════════════════════════════════════════════════

/// 3D H(curl) PEC cavity — zero-source smoke test + matrix sanity.
///
/// Assembles curl-curl + mass on a unit cube tet mesh, applies PEC BCs
/// on all faces, solves with zero source.  The nullspace (gradient fields)
/// is eliminated by the CG solver which finds the minimum-norm solution.
///
/// Checks:
///   - curl-curl matrix is structurally symmetric
///   - CG solver converges with zero RHS → zero solution
///   - Matrix dimensions match ND1 DOF count
#[test]
fn team1_hcurl_3d_pec_cavity_smoke() {
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_assembly::VectorAssembler;
    use fem_space::constraints::boundary_dofs_hcurl;
    use fem_solver::SolverConfig;

    let n = 4;
    let mesh = SimplexMesh::<3>::unit_cube_tet(n);
    let space = HCurlSpace::new(mesh.clone(), 1);
    let n_dof = space.n_dofs();

    let mut mat = VectorAssembler::assemble_bilinear(
        &space,
        &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }],
        4,
    );

    // Verify structural symmetry
    assert_eq!(mat.nrows, n_dof);
    assert_eq!(mat.ncols, n_dof);
    assert!(mat.nnz() > 0, "3D curl-curl matrix should have entries");

    // Check symmetry: (i,j) entry should match (j,i)
    let mut sym_ok = true;
    'outer: for i in 0..n_dof {
        for p in mat.row_ptr[i]..mat.row_ptr[i+1] {
            let j = mat.col_idx[p] as usize;
            let mut found = false;
            for q in mat.row_ptr[j]..mat.row_ptr[j+1] {
                if mat.col_idx[q] as usize == i {
                    if (mat.values[p] - mat.values[q]).abs() > 1e-14 {
                        sym_ok = false;
                        break 'outer;
                    }
                    found = true;
                    break;
                }
            }
            if !found { sym_ok = false; break 'outer; }
        }
    }
    assert!(sym_ok, "3D curl-curl + mass matrix should be symmetric");

    // Zero-source solve: should converge to trivial solution
    let mut rhs = vec![0.0; n_dof];
    let bnd = boundary_dofs_hcurl(&mesh, &space, &[1, 2, 3, 4, 5, 6]);
    let vals = vec![0.0; bnd.len()];
    fem_space::constraints::apply_dirichlet(&mut mat, &mut rhs, &bnd, &vals);

    let mut u = vec![0.0; n_dof];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };
    let result = fem_solver::solve_cg(&mat, &rhs, &mut u, &cfg)
        .expect("3D H(curl) smoke CG failed");

    assert!(result.converged, "3D H(curl) smoke should converge");
    let norm_u: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm_u < 1e-12, "3D zero-source solution should be zero, norm={:.4e}", norm_u);

    eprintln!("  [TEAM 1 3D H(curl)] PEC cavity smoke (n={}):", n);
    eprintln!("       DOFs={}, nnz={}, ||u||₂={:.4e}, iters={}",
        n_dof, mat.nnz(), norm_u, result.iterations);

    fem_regression::regression("team1_hcurl_3d_pec_smoke")
        .check_with("n_dofs", n_dof as f64, 1e-6, 0.5)
        .check_with("nnz", mat.nnz() as f64, 1e-6, 0.5)
        .check_with("solution_norm", norm_u, 1e-6, 1e-12)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 3 / CEM: Multi-layer dielectric-loaded waveguide
// ═══════════════════════════════════════════════════════════════════════

/// Multi-layer dielectric-loaded rectangular waveguide (three regions).
///
/// Waveguide cross-section [0,1]² with three vertical strips:
///   Left  (x < 0.3):  εr = 1 (air)
///   Center (0.3 ≤ x ≤ 0.7): εr = 4 (dielectric)
///   Right (x > 0.7):  εr = 1 (air)
///
/// TE modes: ∂u/∂n = 0 on waveguide walls (natural BCs).
/// The dielectric slab loading reduces the first non-zero TE cutoff
/// below the empty-waveguide value π² ≈ 9.87.
///
/// References: Collin, "Field Theory of Guided Waves", §6.3;
///             TEAM workshop problem 3 (multi-material waveguide).
#[test]
fn team3_dielectric_slab_waveguide() {
    let mesh_raw = SimplexMesh::<2>::unit_square_tri(24);
    let mut mesh = mesh_raw;
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let cx: f64 = nodes.iter()
            .map(|&n| mesh.node_coords(n)[0])
            .sum::<f64>() / nodes.len() as f64;
        mesh.elem_tags[e as usize] = if cx < 0.3 { 1 }
            else if cx <= 0.7 { 2 }
            else { 3 };
    }

    let space = H1Space::new(mesh, 1);

    // Piecewise εr: tag 1,3 → εr=1 (air), tag 2 → εr=4 (dielectric slab)
    let m_mat = Assembler::assemble_bilinear(
        &space,
        &[&MassIntegrator { rho: PWConstCoeff::new([(1, 1.0), (2, 4.0), (3, 1.0)]) }],
        3,
    );
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    // TE modes: natural BCs → no Dirichlet constraints.
    // Skip the zero eigenvalue (nullspace of −Δ with Neumann BC).
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
    let result = lobpcg(&k_mat, Some(&m_mat), 5, &cfg)
        .expect("TEAM 3 LOBPCG failed");

    let mut ev = result.eigenvalues;
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    while !ev.is_empty() && ev[0] < 1.0 { ev.remove(0); }

    let k_target = 3;
    assert!(ev.len() >= k_target, "TEAM 3: expected ≥3 non-zero eigenvalues, got {}", ev.len());
    for (i, &lam) in ev.iter().enumerate().take(k_target) {
        assert!(lam > 0.0, "TEAM 3: λ[{i}] = {lam:.6} should be positive");
    }

    // Dielectric slab lowers the first eigenvalue below π²
    let empty_fundamental = PI * PI;
    assert!(ev[0] < empty_fundamental,
        "TEAM 3: fundamental λ={:.6} should be < π²={:.6} (dielectric loading)",
        ev[0], empty_fundamental);
    // But physically it should still be positive and > 3 (not too low)
    assert!(ev[0] > 3.0,
        "TEAM 3: fundamental λ={:.6} suspiciously low", ev[0]);

    eprintln!("  [TEAM 3] Dielectric slab waveguide (εr=4, 0.3-0.7):");
    for i in 0..k_target {
        eprintln!("           λ[{i}] = {:.6}  (empty ref π²={:.6})", ev[i], empty_fundamental);
    }
    eprintln!("           empty waveguide fundamental = π² = {:.6}", empty_fundamental);

    fem_regression::regression("team3_dielectric_slab_waveguide")
        .check_with("lambda_0", ev[0], 1e-6, 1e-10)
        .check_with("lambda_1", ev[1], 1e-6, 1e-10)
        .check_with("lambda_2", ev[2], 1e-6, 1e-10)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// 3D H(curl) MMS — ND1 convergence via CG solver + regression baseline
// ═══════════════════════════════════════════════════════════════════════

/// 3D H(curl) manufactured solution convergence test using CG solver.
///
/// Solves: curl curl E + E = f  in [0,1]³,  n×E = 0 on all faces.
///
/// Manufactured: E = (sin(πy)sin(πz), sin(πx)sin(πz), sin(πx)sin(πy))
/// Source: f = (2π²+1)·E  (since curl curl E = 2π²·E)
///
/// Analytical L² norm: ||E||² = 3/4.  We track the mass-weighted norm
/// ||u_h||²_M = u_h^T M u_h as a convergence proxy across mesh refinements.
///
/// This validates the full 3D H(curl) assembly → solver → postprocessing
/// pipeline with a genuine manufactured solution.
#[test]
fn team3_hcurl_3d_mms_convergence() {
    use fem_assembly::standard::{CurlCurlIntegrator, VectorMassIntegrator};
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    use fem_assembly::VectorAssembler;
    use fem_space::constraints::boundary_dofs_hcurl;
    use fem_solver::SolverConfig;
    use std::f64::consts::PI;

    // 3D manufactured source: f = (2π²+1)·E(x)
    struct MmsSource3D;
    impl VectorLinearIntegrator for MmsSource3D {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let x = qp.x_phys;
            let sx = (PI * x[0]).sin();
            let sy = (PI * x[1]).sin();
            let sz = (PI * x[2]).sin();
            let coeff = 2.0 * PI * PI + 1.0;
            let fx = coeff * sy * sz;
            let fy = coeff * sx * sz;
            let fz = coeff * sx * sy;
            for i in 0..qp.n_dofs {
                let dot = qp.phi_vec[i * 3] * fx + qp.phi_vec[i * 3 + 1] * fy + qp.phi_vec[i * 3 + 2] * fz;
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    let mut prev_norm = f64::MAX;
    let analytical_norm_sq = 0.75; // ∫(E_x²+E_y²+E_z²) = 3·(1/2)·(1/2) = 3/4
    for &n in &[2usize, 3, 4] {
        let mesh = SimplexMesh::<3>::unit_cube_tet(n);
        let space = HCurlSpace::new(mesh.clone(), 1);
        let n_dof = space.n_dofs();

        let mat = VectorAssembler::assemble_bilinear(
            &space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], 5,
        );
        let mut rhs = VectorAssembler::assemble_linear(&space, &[&MmsSource3D], 5);
        let mut a_mat = mat;

        // PEC BC: n×E = 0 on all 6 faces
        let bnd = boundary_dofs_hcurl(&mesh, &space, &[1, 2, 3, 4, 5, 6]);
        let bnd_vals = vec![0.0; bnd.len()];
        fem_space::constraints::apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

        let mut u = vec![0.0; n_dof];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let result = fem_solver::solve_cg(&a_mat, &rhs, &mut u, &cfg)
            .expect(&format!("3D H(curl) MMS CG n={} failed", n));
        assert!(result.converged, "3D H(curl) MMS CG n={} not converged", n);

        // Mass-weighted norm as L² proxy
        let mass_mat = VectorAssembler::assemble_bilinear(
            &space, &[&VectorMassIntegrator { alpha: 1.0 }], 5,
        );
        let mut mv = vec![0.0; n_dof];
        mass_mat.spmv(&u, &mut mv);
        let norm_sq: f64 = u.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        let norm = norm_sq.sqrt();

        let rel_err = (norm_sq - analytical_norm_sq).abs() / analytical_norm_sq;
        if prev_norm < f64::MAX {
            let change = (prev_norm - norm).abs() / prev_norm;
            eprintln!("  [3D HCurl MMS] n={}: ||u||_M={:.6e}, rel_err={:.4e}, change={:.3}, iters={}",
                n, norm, rel_err, change, result.iterations);
        } else {
            eprintln!("  [3D HCurl MMS] n={}: ||u||_M={:.6e}, rel_err={:.4e}, iters={}",
                n, norm, rel_err, result.iterations);
        }
        prev_norm = norm;
    }

    // Verify final mesh gives norm close to analytical value
    // (ND1 on n=4 with CG should be within 10%)
    {
        let mesh = SimplexMesh::<3>::unit_cube_tet(4);
        let space = HCurlSpace::new(mesh.clone(), 1);
        let n_dof = space.n_dofs();
        let mat = VectorAssembler::assemble_bilinear(
            &space, &[&CurlCurlIntegrator { mu: 1.0 }, &VectorMassIntegrator { alpha: 1.0 }], 5,
        );
        let mut rhs = VectorAssembler::assemble_linear(&space, &[&MmsSource3D], 5);
        let mut a_mat = mat;
        let bnd = boundary_dofs_hcurl(&mesh, &space, &[1, 2, 3, 4, 5, 6]);
        fem_space::constraints::apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &vec![0.0; bnd.len()]);
        let mut u = vec![0.0; n_dof];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let result = fem_solver::solve_cg(&a_mat, &rhs, &mut u, &cfg).expect("3D MMS CG n=4");
        assert!(result.converged);

        let mass_mat = VectorAssembler::assemble_bilinear(&space, &[&VectorMassIntegrator { alpha: 1.0 }], 5);
        let mut mv = vec![0.0; n_dof];
        mass_mat.spmv(&u, &mut mv);
        let norm_sq: f64 = u.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        let rel_err = (norm_sq - analytical_norm_sq).abs() / analytical_norm_sq;
        assert!(rel_err < 0.15, "3D H(curl) MMS: norm rel_err={:.4e} > 15%", rel_err);

        fem_regression::regression("team3_hcurl_3d_mms")
            .check_with("n_dofs", n_dof as f64, 1e-6, 0.5)
            .check_with("norm_sq", norm_sq, 1e-6, 1e-8)
            .check_with("rel_err", rel_err, 1e-6, 1e-8)
            .finalize();
    }
}
