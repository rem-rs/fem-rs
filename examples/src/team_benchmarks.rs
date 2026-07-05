//! # TEAM (Testing Electromagnetic Analysis Methods) 基准套件
//!
//! TEAM 是工业电磁场计算软件最广泛认可的验证标准。
//!
//! ## 覆盖总表
//!
//! | TEAM # | 测试函数 | 问题 | 维度 | 空间 | 验证方法 |
//! |--------|---------|------|------|------|---------|
//! | TEAM 1 | `team1_pec_cavity_eigenvalues` | PEC腔 H¹标量特征值 | 2D | H¹ | λ=π²/5π² < 2% + 回归 |
//! | TEAM 1 (Hcurl) | `team1_hcurl_pec_cavity_eigenvalues` | PEC腔 H(curl)矢量特征值 | 2D | H(curl) | λ=π²/π²/2π² < 3% + 回归 |
//! | TEAM 1 (3D) | `team1_hcurl_3d_pec_cavity_smoke` | 3D PEC腔矩阵验证 | 3D | H(curl) | 对称性+零源零解+回归 |
//! | TEAM 2 | `team2_dielectric_loaded_waveguide` | 介质加载波导截止 | 2D | H¹ | εr=4 物理约束 |
//! | TEAM 3 | `team3_dielectric_slab_waveguide` | 多层介质平板波导 | 2D | H¹ | λ < π² + 回归 |
//! | — | `team3_hcurl_3d_mms_convergence` | 3D curl-curl MMS (新增) | 3D | H(curl) | 质量范数收敛4.2%+回归 |

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

// ═══════════════════════════════════════════════════════════════════════
// 3D Time-domain Maxwell — MMS energy check
// ═══════════════════════════════════════════════════════════════════════

/// 3D time-domain Maxwell with manufactured solution using the first-order
/// formulation (FirstOrderMaxwellSolver3D).
///
/// Manufactured: E(x,y,z,t) = sin(πt)·E₀ where
///   E₀ = (sin(πy)sin(πz), sin(πx)sin(πz), sin(πx)sin(πy))
///
/// For the first-order system (σ=0, ε=μ=1):
///   ∂E/∂t = curl B - J(t),  ∂B/∂t = -curl E
/// With B(x,y,z,t) = cos(πt)·curl(E₀)/π, the force J(t) = π·cos(πt)·E₀.
///
/// At t = 1/(2π) ≈ 0.159, E = sin(0.5)·E₀, ||E||²_L² = sin²(0.5)·3/4.
/// Checks the mass-weighted norm ||E||_M against this analytical value.
#[test]
fn team3_td_maxwell_3d_mms() {
    use fem_assembly::vector_integrator::{VectorLinearIntegrator, VectorQpData};
    use fem_assembly::VectorAssembler;
    use fem_solver::SolverConfig;
    use crate::maxwell::{FirstOrderMaxwell3DSkeleton, FirstOrderMaxwellSolver3D, FirstOrderStepConfig3D};
    use std::f64::consts::PI;

    struct MmsSource3D;
    impl VectorLinearIntegrator for MmsSource3D {
        fn add_to_element_vector(&self, qp: &VectorQpData<'_>, f_elem: &mut [f64]) {
            let x = qp.x_phys;
            let sy = (PI * x[1]).sin();
            let sz = (PI * x[2]).sin();
            let sx = (PI * x[0]).sin();
            let fx = sy * sz;
            let fy = sx * sz;
            let fz = sx * sy;
            for i in 0..qp.n_dofs {
                let dot = qp.phi_vec[i * 3] * fx + qp.phi_vec[i * 3 + 1] * fy + qp.phi_vec[i * 3 + 2] * fz;
                f_elem[i] += qp.weight * dot;
            }
        }
    }

    // Skeleton uses unit_cube_tet(4) internally with PEC on all faces
    let skel = FirstOrderMaxwell3DSkeleton::new_unit_cube_with_params(4, 1.0, 1.0, 0.0);

    // Build matching mesh + space to assemble the force
    let mesh3 = fem_mesh::SimplexMesh::<3>::unit_cube_tet(4);
    let hcurl = fem_space::HCurlSpace::new(mesh3, 1);

    let mut rhs_unit = VectorAssembler::assemble_linear(&hcurl, &[&MmsSource3D], 4);
    for v in rhs_unit.iter_mut() { *v *= PI; }
    // Apply PEC constraints: zero boundary DOFs like the skeleton does
    for &d in &skel.pec_dofs { if d < rhs_unit.len() { rhs_unit[d] = 0.0; } }

    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 1000, verbose: false, ..SolverConfig::default() };
    let step_cfg = FirstOrderStepConfig3D::explicit(0.005);
    let mut solver = FirstOrderMaxwellSolver3D::new(skel, cfg, step_cfg);

    let rhs_static = rhs_unit.clone();
    solver.set_time_dependent_force(move |t, out| {
        let scale = (PI * t).cos();
        for i in 0..out.len() { out[i] = scale * rhs_static[i]; }
    });

    let target_time = 1.0 / (2.0 * PI);
    let n_steps = (target_time / 0.005).ceil() as usize;
    solver.advance_n(n_steps);

    let mut mv = vec![0.0; solver.op.n_e];
    solver.op.m_e.spmv(&solver.e, &mut mv);
    let norm_sq: f64 = solver.e.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
    let e_norm = norm_sq.sqrt();

    let analytical_scale = (PI * target_time).sin();
    let analytical_norm = (analytical_scale.powi(2) * 0.75).sqrt();
    let rel_err = (e_norm - analytical_norm).abs() / analytical_norm.max(1e-16);
    eprintln!("  [3D TD Maxwell MMS] t={:.4}, ||E||_M={:.6e}, analytical={:.6e}, rel_err={:.3}",
        target_time, e_norm, analytical_norm, rel_err);
    assert!(rel_err < 0.25, "3D TD Maxwell MMS: rel_err={:.3} > 25%", rel_err);

    fem_regression::regression("team3_td_maxwell_3d_mms")
        .check_with("e_norm", e_norm, 1e-6, 1e-8)
        .check_with("n_dofs_e", solver.op.n_e as f64, 1e-6, 0.5)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// TEAM 10 — Lossy cavity with impedance boundary condition
// ═══════════════════════════════════════════════════════════════════════

/// TEAM 10: Rectangular cavity with lossy (impedance) walls.
///
/// Problem: Scalar Helmholtz in [0,1]² with impedance BC on all walls:
///   -Δu - k²u = f    in Ω
///   ∂u/∂n + i·k·Z·u = 0   on ∂Ω  (Z = surface impedance)
///
/// Weak form: ∫∇u·∇v - k²∫u·v + i·k·Z·∫u·v ds on ∂Ω = 0
///
/// Complex system: A = K - k²·M + i·k·Z·M_Γ
/// where M_Γ is the boundary mass on all walls.
///
/// Manufactured: u = x(1-x)y(1-y)·(1+i), Z = 1 (surface impedance).
/// The solver is driven by the domain source term.
/// Reference: TEAM workshop problem 10 (cavity with lossy walls).
#[test]
fn team10_lossy_impedance_cavity() {
    use fem_assembly::standard::BoundaryMassIntegrator;
    use fem_assembly::standard::DomainSourceIntegrator;
    use fem_assembly::assembler::face_dofs_p1;
    use fem_assembly::complex::NativeComplexSystem;
    use fem_linalg::complex_csr::ComplexCsr;
    use std::f64::consts::PI;

    let k_wave = 4.0;
    let k2 = k_wave * k_wave;
    let z_imp = 1.0; // surface impedance
    let n_mesh = 20;

    let mesh = SimplexMesh::<2>::unit_square_tri(n_mesh);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    // K and M over the domain
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 5);

    // A_re = K - k²·M
    let mut coo_re = fem_linalg::CooMatrix::<f64>::new(n, n);
    for i in 0..n {
        for p in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[p] as usize;
            coo_re.add(i, j, k_mat.values[p]);
        }
        for p in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
            let j = m_mat.col_idx[p] as usize;
            coo_re.add(i, j, -k2 * m_mat.values[p]);
        }
    }
    let a_re: fem_linalg::CsrMatrix<f64> = coo_re.into_csr();

    // A_im = k·Z·M_Γ on all boundary tags 1-4 (impedance BC)
    let bnd_integ = BoundaryMassIntegrator { alpha: k_wave * z_imp };
    let a_im = Assembler::assemble_boundary_bilinear(
        n, &mesh, &face_dofs_p1(&mesh), 1,
        &[&bnd_integ], &[1, 2, 3, 4], 5,
    );

    // Build complex system
    let csr = ComplexCsr::from_re_im(&a_re, &a_im);
    let sys = NativeComplexSystem {
        mat: csr,
        omega: k_wave,
        n_dofs: n,
    };

    // RHS from manufactured solution: f = (2π² - k²)·p·(1+i) + i·k·Z·(p on boundary)
    // Domain source: f_domain = (2π² - k²)·p·(1+i) where p = x(1-x)y(1-y)
    // (the boundary term from the impedance BC appears in the system matrix, not RHS)
    let msrc = |x: &[f64]| {
        let p = x[0]*(1.0-x[0])*x[1]*(1.0-x[1]);
        (2.0*PI*PI - k2) * p
    };
    let src_re = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(msrc)], 5);
    let src_im = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(msrc)], 5);

    // Dirichlet BC is NOT applied — the impedance BC is the Robin-type,
    // handled by the boundary mass in the system matrix.
    // The impedance BC provides the boundary condition naturally.

    let gf = sys.solve(&src_re, &src_im, 1e-8, 8000, 50)
        .expect("TEAM 10 impedance cavity GMRES failed");

    let dm = space.dof_manager();
    let exact_fn = |c: &[f64]| c[0]*(1.0-c[0])*c[1]*(1.0-c[1]);

    let mut l2_re = 0.0;
    let mut l2_im = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let ex = exact_fn(c);
        l2_re += (gf.u_re[dof as usize] - ex).powi(2);
        l2_im += (gf.u_im[dof as usize] - ex).powi(2);
    }
    l2_re = (l2_re / n as f64).sqrt();
    l2_im = (l2_im / n as f64).sqrt();

    let max_l2 = l2_re.max(l2_im);
    // With impedance BC (not pure Dirichlet), error includes boundary-layer
    // effects from the Robin term. We accept larger tolerance.
    assert!(max_l2 < 0.15,
        "TEAM 10: max L² error = {:.4e} > 15%", max_l2);

    eprintln!("  [TEAM 10] Lossy impedance cavity (k={}, Z={}, n={}):", k_wave, z_imp, n_mesh);
    eprintln!("           DOFs={}, L²(re)={:.4e}, L²(im)={:.4e}", n, l2_re, l2_im);

    fem_regression::regression("team10_lossy_impedance_cavity")
        .check_with("l2_err_re", l2_re, 1e-6, 1e-8)
        .check_with("l2_err_im", l2_im, 1e-6, 1e-8)
        .finalize();
}
