//! NAFEMS-style EM benchmarks for fem-rs.
//!
//! Validates electromagnetic FEM computations against analytical solutions:
//!
//! | Benchmark | Problem | Reference |
//! |-----------|---------|-----------|
//! | TE waveguide | 2-D Helmholtz, ∂u/∂n=0 | λ = π²(m²+n²), m,n≥0 |
//! | TM waveguide | 2-D Helmholtz, u=0 | λ = π²(m²+n²), m,n≥1 |
//! | Dielectric-loaded cavity | 2-D Helmholtz, εr(x) piecewise | Frequency ratio |

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
    H1Space,
    constraints::boundary_dofs,
};

// ─── Helpers ────────────────────────────────────────────────────────────

/// Assign element tags 1 (left half) or 2 (right half) based on x-coordinate.
fn tag_mesh_by_x(mut mesh: SimplexMesh<2>, split_x: f64) -> SimplexMesh<2> {
    for e in 0..mesh.n_elements() as u32 {
        let nodes = mesh.element_nodes(e);
        let mut cx = 0.0;
        for &n in nodes {
            let c = mesh.node_coords(n);
            cx += c[0];
        }
        cx /= nodes.len() as f64;
        mesh.elem_tags[e as usize] = if cx < split_x { 1 } else { 2 };
    }
    mesh
}

/// Sort eigenvalues and keep the smallest `k`.
fn extract_eigenvalues(result: &fem_solver::EigenResult, k: usize) -> Vec<f64> {
    let mut ev = result.eigenvalues.clone();
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev.truncate(k);
    ev
}

/// Extract the free-DOF submatrix (rows/cols NOT in `constrained_dofs`).
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

/// Analytical TE waveguide eigenvalues: λ = π²(m²+n²) for m,n ≥ 0, not both 0.
fn te_analytical(k: usize) -> Vec<f64> {
    let mut ev: Vec<f64> = Vec::new();
    for m in 0..10i32 {
        for n in 0..10i32 {
            if m == 0 && n == 0 { continue; }
            ev.push(PI * PI * ((m * m + n * n) as f64));
        }
    }
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev.truncate(k);
    ev
}

/// Analytical TM waveguide eigenvalues: λ = π²(m²+n²) for m,n ≥ 1.
fn tm_analytical(k: usize) -> Vec<f64> {
    let mut ev: Vec<f64> = Vec::new();
    for m in 1..=10i32 {
        for n in 1..=10i32 {
            ev.push(PI * PI * ((m * m + n * n) as f64));
        }
    }
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev.truncate(k);
    ev
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark EM1: TE waveguide cutoff
// ═══════════════════════════════════════════════════════════════════════

/// TE modes: -Δu = λ u, ∂u/∂n = 0 (natural BCs).
///
/// The K matrix has a nullspace (constant function, λ = 0).
/// We skip the zero mode by requesting k+1 eigenvalues and discarding
/// the first (near-zero) one.
#[test]
fn em_te_waveguide_cutoff() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let space = H1Space::new(mesh, 1);

    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

    // Natural BC → request k+1, skip first (zero) eigenvalue
    let k_target = 5;
    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
    let result = lobpcg(&k_mat, Some(&m_mat), k_target + 1, &cfg)
        .expect("TE waveguide LOBPCG failed");

    let mut ev = result.eigenvalues;
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Discard the zero (near-zero) eigenvalue
    while !ev.is_empty() && ev[0] < 1.0 { ev.remove(0); }
    let ev = &ev[..k_target.min(ev.len())];

    let exact = te_analytical(k_target);

    for i in 0..k_target.min(ev.len()) {
        let err = (ev[i] - exact[i]).abs() / exact[i].max(1.0);
        assert!(err < 0.03,
            "TE mode {i}: computed λ={:.6}, exact λ={:.6}, rel_err={:.3}",
            ev[i], exact[i], err);
    }
    eprintln!("  [EM] TE waveguide cutoffs (n=16):");
    for i in 0..k_target.min(ev.len()) {
        eprintln!("       λ[{i}] = {:.6}  (exact {:.6})", ev[i], exact[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark EM2: TM waveguide cutoff
// ═══════════════════════════════════════════════════════════════════════

/// TM modes: -Δu = λ u, u = 0 on boundary.
///
/// Use free-DOF submatrix extraction (same pattern as ex13_laplacian_eigen)
/// to impose Dirichlet BCs without breaking matrix symmetry.
#[test]
fn em_tm_waveguide_cutoff() {
    let mesh = SimplexMesh::<2>::unit_square_tri(16);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

    // Build free-DOF index set (interior nodes)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
    let free: Vec<usize> = (0..n).filter(|&i| !bnd_set.contains(&(i as u32))).collect();

    let k_free = free_submatrix(&k_mat, &free);
    let m_free = free_submatrix(&m_mat, &free);

    let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
    let result = lobpcg(&k_free, Some(&m_free), 3, &cfg)
        .expect("TM waveguide LOBPCG failed");

    let ev = extract_eigenvalues(&result, 3);
    let exact = tm_analytical(3);

    // Allow 3% error for P1 on n=16
    for i in 0..3 {
        let err = (ev[i] - exact[i]).abs() / exact[i].max(1.0);
        assert!(err < 0.03,
            "TM mode {i}: computed λ={:.6}, exact λ={:.6}, rel_err={:.3}",
            ev[i], exact[i], err);
    }
    eprintln!("  [EM] TM waveguide cutoffs (n=16):");
    for i in 0..3 {
        eprintln!("       λ[{i}] = {:.6}  (exact {:.6})", ev[i], exact[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark EM3: Dielectric-loaded cavity
// ═══════════════════════════════════════════════════════════════════════

/// Cavity half-filled with dielectric (εr = 4).
///
/// TM_z mode: -ΔE_z = ω² εr E_z, E_z = 0 on PEC walls.
/// Left half: εr=1, right half: εr=4.
///
/// Dielectric loading lowers the resonant frequency.
#[test]
fn em_dielectric_loaded_cavity() {
    let mesh_raw = SimplexMesh::<2>::unit_square_tri(16);
    let mesh = tag_mesh_by_x(mesh_raw, 0.5);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    // Stiffness uses constant κ=1, mass uses εr(x) via PWConstCoeff
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
    let m_mat = Assembler::assemble_bilinear(
        &space, &[&MassIntegrator { rho: PWConstCoeff::new([(1, 1.0), (2, 4.0)]) }], 3
    );

    // Free-DOF extraction for Dirichlet BC (E_z = 0 on PEC walls)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
    let bnd_set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
    let free: Vec<usize> = (0..n).filter(|&i| !bnd_set.contains(&(i as u32))).collect();

    let k_free = free_submatrix(&k_mat, &free);
    let m_free = free_submatrix(&m_mat, &free);

    let cfg = LobpcgConfig { max_iter: 800, tol: 1e-8, verbose: false };
    let result = lobpcg(&k_free, Some(&m_free), 3, &cfg)
        .expect("dielectric cavity LOBPCG failed");

    let ev = extract_eigenvalues(&result, 3);
    let vacuum_fundamental = 2.0 * PI * PI;

    for (i, &lam) in ev.iter().enumerate() {
        assert!(lam > 0.0, "eigenvalue {i} should be positive: {:.6}", lam);
    }
    assert!(ev[0] < vacuum_fundamental,
        "dielectric loading should reduce fundamental: {:.6} vs {:.6}",
        ev[0], vacuum_fundamental);
    assert!(ev[0] > 0.1 * vacuum_fundamental,
        "physically unreasonable fundamental: {:.6}", ev[0]);

    eprintln!("  [EM] dielectric-loaded cavity (εr=4, half-fill, n=16):");
    for i in 0..3 {
        eprintln!("       λ[{i}] = {:.6}  (vacuum fund. = {vacuum_fundamental:.6})", ev[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Benchmark EM4: Cavity eigenvalue convergence
// ═══════════════════════════════════════════════════════════════════════

/// Verify that TM cavity eigenvalues converge to analytical values
/// as the mesh is refined (free-DOF extraction approach).
#[test]
fn em_cavity_eigenvalue_convergence() {
    let mut prev_err: f64 = f64::MAX;

    for &n in &[8, 12, 16] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh.clone(), 1);
        let n_dofs = space.n_dofs();

        let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);
        let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 3);

        // Free-DOF extraction
        let dm = space.dof_manager();
        let bnd = boundary_dofs(space.mesh(), dm, &[1, 2, 3, 4]);
        let bnd_set: std::collections::HashSet<u32> = bnd.iter().cloned().collect();
        let free: Vec<usize> = (0..n_dofs).filter(|&i| !bnd_set.contains(&(i as u32))).collect();

        let k_free = free_submatrix(&k_mat, &free);
        let m_free = free_submatrix(&m_mat, &free);

        let cfg = LobpcgConfig { max_iter: 500, tol: 1e-8, verbose: false };
        let result = lobpcg(&k_free, Some(&m_free), 3, &cfg)
            .expect("cavity eigenvalue LOBPCG failed");

        let ev = extract_eigenvalues(&result, 3);
        let exact = tm_analytical(3);
        let max_err = (0..3).map(|i| (ev[i] - exact[i]).abs() / exact[i].max(1.0))
            .fold(0.0_f64, f64::max);

        assert!(result.converged, "n={}: LOBPCG should converge", n);
        assert!(max_err < prev_err,
            "n={}: error increased: prev={:.3e}, current={:.3e}",
            n, prev_err, max_err);
        eprintln!("  [EM] cavity convergence n={}: max_rel_err={:.3e}", n, max_err);
        prev_err = max_err;
    }
}
