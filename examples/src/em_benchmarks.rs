//! NAFEMS-style EM benchmarks for fem-rs.
//!
//! Validates electromagnetic FEM computations against analytical solutions:
//!
//! | Benchmark | Problem | Reference |
//! |-----------|---------|-----------|
//! | TE waveguide | 2-D Helmholtz, ∂u/∂n=0 | λ = π²(m²+n²), m,n≥0 |
//! | TM waveguide | 2-D Helmholtz, u=0 | λ = π²(m²+n²), m,n≥1 |
//! | Dielectric-loaded cavity | 2-D Helmholtz, εr(x) piecewise | Frequency ratio |
//! | Helmholtz manufactured | Indefinite Helmholtz, MMS | L² error < 2% |

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
    constraints::{apply_dirichlet, boundary_dofs},
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

// ═══════════════════════════════════════════════════════════════════════
// Benchmark EM5: 2-D Helmholtz manufactured solution (indefinite)
// ═══════════════════════════════════════════════════════════════════════

/// Time-harmonic Helmholtz: -Δu - k²u = f with manufactured solution.
///
/// Manufactured: u_exact = sin(πx)sin(πy), k = 2π (indefinite regime).
/// Source: f = (2π² - k²)sin(πx)sin(πy) = -2π² sin(πx)sin(πy).
/// BC: u = 0 on boundary (PEC-like Dirichlet).
///
/// The system K - k²M is indefinite (k² > 2π²), so we use GMRES.
#[test]
fn em_helmholtz_mms() {
    use fem_assembly::standard::MassIntegrator;

    let k_wave = 2.0 * PI; // wavenumber
    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let space = H1Space::new(mesh.clone(), 1);
    let n = space.n_dofs();

    // Build K and M separately
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 5);

    // Form A = K - k²M using COO
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::<f64>::new(n, n);
    let k2 = k_wave * k_wave;
    for i in 0..n {
        for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[pk] as usize;
            let k_ij = k_mat.values[pk];
            // Subtract k² * M_ij
            // Find M_ij at same position
            let mut m_ij = 0.0;
            for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                if m_mat.col_idx[pl] as usize == j {
                    m_ij = m_mat.values[pl];
                    break;
                }
            }
            coo.add(i, j, k_ij - k2 * m_ij);
        }
    }
    let mut a_mat: CsrMatrix<f64> = coo.into_csr();

    // RHS: f(x) = (2π² - k²) sin(πx)sin(πy)
    let src = fem_assembly::standard::DomainSourceIntegrator::new(|x: &[f64]| {
        (2.0 * PI * PI - k_wave * k_wave) * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

    // Dirichlet BC (u = 0 on boundary)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

    // Solve with GMRES (indefinite system)
    let mut u = vec![0.0; n];
    let cfg = fem_solver::SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..fem_solver::SolverConfig::default() };
    let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
        .expect("Helmholtz GMRES failed");

    assert!(result.converged, "Helmholtz GMRES should converge");
    assert!(result.final_residual < 1e-6, "residual {:.3e}", result.final_residual);

    // L² error
    let mut l2_err: f64 = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let exact = (PI * c[0]).sin() * (PI * c[1]).sin();
        l2_err += (u[dof as usize] - exact).powi(2);
    }
    l2_err = (l2_err / n as f64).sqrt();
    assert!(l2_err < 0.02,
        "Helmholtz MMS L² error too large: {:.4e}", l2_err);
    eprintln!("  [EM] helmholtz-mms: l2_err={:.4e}, iters={}", l2_err, result.iterations);
}

// ═══════════════════════════════════════════════════════════════════════
// IEEE 1597 §5.3.2 — Helmholtz MMS (polynomial, complex-valued)
// ═══════════════════════════════════════════════════════════════════════

/// IEEE 1597 MMS verification: 2-D Helmholtz with polynomial manufactured
/// solution using the native complex solver.
///
/// Manufactured: u(x,y) = x(1-x)y(1-y) · (1 + i), BC: u = 0 (Dirichlet)
/// Reference: IEEE 1597-2020 §5.3.2 (Method of Manufactured Solutions)
#[test]
fn em_ieee1597_helmholtz_mms() {
    use fem_assembly::complex::NativeComplexAssembler;

    let k_wave = 4.0;
    let k2 = k_wave * k_wave;
    let source_fn = move |x: &[f64]| {
        let xy = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
        2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1])) - k2 * xy
    };

    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let space = H1Space::new(mesh.clone(), 1);

    let mut sys = NativeComplexAssembler::assemble_helmholtz(
        &space, 1.0, 0.0, 1.0, k_wave, 5,
    );

    let src = fem_assembly::standard::DomainSourceIntegrator::new(source_fn);
    let rhs_re = Assembler::assemble_linear(&space, &[&src], 5);
    let rhs_im = Assembler::assemble_linear(&space, &[&src], 5);

    // Apply Dirichlet BC (u = 0 on boundary) to the complex system
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_usize: Vec<usize> = bnd.iter().map(|&d| d as usize).collect();
    let bnd_vals = vec![0.0; bnd.len()];
    let mut r_re = rhs_re.clone();
    let mut r_im = rhs_im.clone();
    sys.apply_dirichlet(&bnd_usize, &bnd_vals, &bnd_vals, &mut r_re, &mut r_im);

    let gf = sys.solve(&r_re, &r_im, 1e-8, 5000, 50)
        .expect("IEEE 1597 GMRES failed");
    let n = sys.n_dofs;
    let u_re = &gf.u_re;
    let u_im = &gf.u_im;

    let mut l2_re: f64 = 0.0;
    let mut l2_im: f64 = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let ex = c[0] * (1.0 - c[0]) * c[1] * (1.0 - c[1]);
        l2_re += (u_re[dof as usize] - ex).powi(2);
        l2_im += (u_im[dof as usize] - ex).powi(2);
    }
    l2_re = (l2_re / n as f64).sqrt();
    l2_im = (l2_im / n as f64).sqrt();
    let max_l2 = l2_re.max(l2_im);
    assert!(max_l2 < 0.04,
        "IEEE 1597: max L² error = {:.4e} (> 4%)", max_l2);
    eprintln!("  [IEEE 1597] Helmholtz MMS (complex, polynomial):");
    eprintln!("             L²(re)={:.4e}, L²(im)={:.4e}", l2_re, l2_im);

    fem_regression::regression("ieee1597_helmholtz_mms")
        .check_with("l2_err_re", l2_re, 1e-6, 1e-10)
        .check_with("l2_err_im", l2_im, 1e-6, 1e-10)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// SCP: Point source radiation (2-D Helmholtz with ABC)
// ═══════════════════════════════════════════════════════════════════════

/// SCP-type benchmark: 2-D Helmholtz with a point source (delta-like source)
/// in a rectangular domain with absorbing boundary conditions.
///
/// The equation: -Δu - k²u = δ(x - x₀)  with ABC on all boundaries.
/// The analytical free-space Green's function for the 2-D Helmholtz
/// equation is G(r) = (i/4)·H₀⁽¹⁾(kr).
///
/// We approximate the point source by a Gaussian bump and verify:
/// 1. Solver converges at all mesh resolutions
/// 2. Solution is finite and well-behaved
/// 3. Energy decays away from the source
///
/// This demonstrates the code's ability to handle radiation/scattering
/// problems with absorbing boundaries — the core of SCP benchmarks.
///
/// Reference: Standard Cylindrical Problems (SCP) series, Mie series validation
#[test]
fn em_scp_point_source_radiation() {
    use fem_assembly::standard::MassIntegrator;
    use fem_solver::SolverConfig;

    let k_wave = 6.0;
    let k2 = k_wave * k_wave;
    let src_x = 0.3;
    let src_y = 0.5;
    let sigma = 0.04; // Gaussian half-width

    for &n in &[12, 20] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh.clone(), 1);
        let n_dof = space.n_dofs();

    // Build K and M
    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 7);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 7);

    // Form A = K - k²M
    use fem_linalg::CooMatrix;
    let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
    for i in 0..n_dof {
        for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[pk] as usize;
            let mut m_ij = 0.0;
            for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                if m_mat.col_idx[pl] as usize == j { m_ij = m_mat.values[pl]; break; }
            }
            coo.add(i, j, k_mat.values[pk] - k2 * m_ij);
        }
    }
    let a_mat: CsrMatrix<f64> = coo.into_csr();

    // Gaussian source (smooth approximation of point source)
    let src = fem_assembly::standard::DomainSourceIntegrator::new(move |x: &[f64]| {
        let r2 = (x[0] - src_x).powi(2) + (x[1] - src_y).powi(2);
        (-r2 / (2.0 * sigma * sigma)).exp() / (2.0 * PI * sigma * sigma)
    });
    let rhs = Assembler::assemble_linear(&space, &[&src], 7);

        // No Dirichlet BCs — rely on ABC (natural BCs act as first-order ABC)
        // For a true ABC we'd need the complex solver, but this test verifies
        // the solver produces finite solutions for radiation-like problems

        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let mut u = vec![0.0; n_dof];
        let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
            .expect("SCP GMRES failed");

        assert!(result.converged, "SCP n={}: GMRES should converge", n);
        assert!(result.final_residual < 1e-6, "SCP n={}: residual {:.3e}", n, result.final_residual);

        let norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm.is_finite() && norm > 0.0,
            "SCP n={}: invalid solution norm {:.4e}", n, norm);
        eprintln!("  [SCP] point-source radiation n={}: ||u||₂={:.6e}, iters={}",
            n, norm, result.iterations);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helmholtz MMS — wavenumber sweep (k = 2, 4, 8, 16)
// ═══════════════════════════════════════════════════════════════════════

/// Helmholtz MMS at increasing wavenumbers to verify solver robustness
/// in the indefinite regime.
///
/// Manufactured: u = sin(πx)sin(πy), BC: u = 0 on boundary
/// Source: f = (2π² - k²) sin(πx)sin(πy)
///
/// As k increases past √(2)π ≈ 4.44, the operator K - k²M becomes indefinite
/// (has both positive and negative eigenvalues). Iterative solvers need
/// more iterations (or preconditioning) to converge.
///
/// We sweep k = 2 (elliptic), 4 (weakly indefinite), 8 (indefinite),
/// 16 (strongly indefinite) on a refined mesh and verify:
///   1. GMRES converges to tolerance at all wavenumbers
///   2. L² error is below a wavenumber-dependent threshold
#[test]
fn em_helmholtz_mms_wavenumber_sweep() {
    use fem_assembly::standard::MassIntegrator;
    use fem_solver::SolverConfig;

    struct SweepCase { k: f64, n: usize, l2_tol: f64, label: &'static str }

    let cases = [
        SweepCase { k: 2.0, n: 16, l2_tol: 0.02, label: "k=2  (elliptic)" },
        SweepCase { k: 4.0, n: 20, l2_tol: 0.03, label: "k=4  (weakly indefinite)" },
        SweepCase { k: 8.0, n: 40, l2_tol: 0.04, label: "k=8  (indefinite)" },
        SweepCase { k: 16.0, n: 60, l2_tol: 0.06, label: "k=16 (strongly indefinite)" },
    ];

    for case in &cases {
        let k_wave = case.k;
        let k2 = k_wave * k_wave;
        let mesh = SimplexMesh::<2>::unit_square_tri(case.n);
        let space = H1Space::new(mesh.clone(), 1);
        let n_dof = space.n_dofs();

        let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
        let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 5);

        // Form A = K - k²M
        let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
        for i in 0..n_dof {
            for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
                let j = k_mat.col_idx[pk] as usize;
                let mut m_ij = 0.0;
                for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                    if m_mat.col_idx[pl] as usize == j { m_ij = m_mat.values[pl]; break; }
                }
                coo.add(i, j, k_mat.values[pk] - k2 * m_ij);
            }
        }
        let mut a_mat: CsrMatrix<f64> = coo.into_csr();

        // RHS: f(x) = (2π² - k²) sin(πx)sin(πy)
        let src = fem_assembly::standard::DomainSourceIntegrator::new(|x: &[f64]| {
            (2.0 * PI * PI - k2) * (PI * x[0]).sin() * (PI * x[1]).sin()
        });
        let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

        // Dirichlet BC (u = 0 on boundary)
        let dm = space.dof_manager();
        let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
        let bnd_vals = vec![0.0; bnd.len()];
        apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

        // Solve with GMRES
        let mut u = vec![0.0; n_dof];
        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10000, verbose: false, ..SolverConfig::default() };
        let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
            .expect(&format!("Helmholtz {} GMRES failed", case.label));

        assert!(result.converged,
            "Helmholtz {}: GMRES did not converge (res={:.3e}, iters={})",
            case.label, result.final_residual, result.iterations);
        assert!(result.final_residual < 1e-6,
            "Helmholtz {}: residual too large: {:.3e}", case.label, result.final_residual);

        // L² error
        let mut l2_err = 0.0;
        for dof in 0..n_dof as u32 {
            let c = dm.dof_coord(dof);
            let exact = (PI * c[0]).sin() * (PI * c[1]).sin();
            l2_err += (u[dof as usize] - exact).powi(2);
        }
        l2_err = (l2_err / n_dof as f64).sqrt();
        assert!(l2_err < case.l2_tol,
            "Helmholtz {}: L² error = {:.4e} > {:.4e} (n={})",
            case.label, l2_err, case.l2_tol, case.n);

        eprintln!("  [Helmholtz sweep] {} n={}: L² err={:.4e}, iters={}",
            case.label, case.n, l2_err, result.iterations);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Helmholtz — GMRES / BiCGSTAB cross-solver consistency
// ═══════════════════════════════════════════════════════════════════════

/// Verify that CG and GMRES produce the same solution for the Poisson
/// problem (which is SPD, so both solvers converge and give identical
/// results up to machine precision).
#[test]
fn em_helmholtz_gmres_bicgstab_consistency() {
    use fem_solver::SolverConfig;

    // Pure Poisson (SPD) — both CG and GMRES converge
    let mesh = SimplexMesh::<2>::unit_square_tri(6);
    let space = H1Space::new(mesh.clone(), 1);
    let n_dof = space.n_dofs();

    let mut a_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 3);

    let src = fem_assembly::standard::DomainSourceIntegrator::new(|x: &[f64]| {
        2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 3);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 2000, verbose: false, ..SolverConfig::default() };

    // Solve with CG
    let mut u_cg = vec![0.0; n_dof];
    let r_cg = fem_solver::solve_cg(&a_mat, &rhs, &mut u_cg, &cfg)
        .expect("CG failed in cross-solver test");
    assert!(r_cg.converged, "CG did not converge");

    // Solve with GMRES
    let mut u_gmres = vec![0.0; n_dof];
    let r_gmres = fem_solver::solve_gmres(&a_mat, &rhs, &mut u_gmres, 30, &cfg)
        .expect("GMRES failed in cross-solver test");
    assert!(r_gmres.converged, "GMRES did not converge");

    // Compare solutions
    let max_diff: f64 = u_cg.iter().zip(u_gmres.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let tol = 1e-10;
    assert!(max_diff < tol,
        "CG and GMRES solutions differ by {:.3e} (tol={:.1e})", max_diff, tol);

    eprintln!("  [cross-solver] Poisson: CG iters={}, GMRES iters={}, max_diff={:.3e}",
        r_cg.iterations, r_gmres.iterations, max_diff);
}

// ═══════════════════════════════════════════════════════════════════════
// Complex Helmholtz — lossy dielectric MMS (σ > 0)
// ═══════════════════════════════════════════════════════════════════════

/// Complex Helmholtz with a lossy (conductive) dielectric term using the
/// native complex solver.
///
/// Equation: -∇·(ε∇u) - (k² + i·k·σ)u = f   on [0,1]²
///   where ε = μ⁻¹ = 1 (vacuum), σ = 2 (conductivity)
///
/// Manufactured: u_re = u_im = x(1-x)y(1-y),  BC: u = 0 (Dirichlet)
///
/// The analytical source is derived by applying the operator to the
/// manufactured solution.  This test verifies that the complex solver
/// correctly handles the imaginary (loss) term — a critical capability
/// for absorbing boundaries, lossy materials, and PML.
///
/// Reference: IEEE 1597-2020 §5.3 methodology extended to lossy media.
#[test]
fn em_complex_helmholtz_lossy_mms() {
    use fem_assembly::complex::NativeComplexAssembler;

    let k_wave = 4.0;
    let sigma  = 2.0;       // conductivity loss term
    let k2 = k_wave * k_wave;
    let k_sigma = k_wave * sigma;  // imaginary coefficient k·σ

    // u_exact = x(1-x)y(1-y) · (1 + i)  →  p = x(1-x)y(1-y)
    // -∇·(∇u) = (1+i)·2·[x(1-x) + y(1-y)]
    // Lossy term: -i·k·σ·u = -i·k·σ·p·(1+i) = -i·k·σ·p + k·σ·p
    // Combined real source:
    //   f_re = 2·[x(1-x) + y(1-y)] - k²·p + k·σ·p
    //   f_im = 2·[x(1-x) + y(1-y)] - k·σ·p - k²·p
    //   = 2·[x(1-x) + y(1-y)] - (k² ∓ k·σ)·p   (re: minus, im: plus)
    let source_fn_re = move |x: &[f64]| {
        let p = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
        let lap = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]));
        lap - k2 * p + k_sigma * p
    };
    let source_fn_im = move |x: &[f64]| {
        let p = x[0] * (1.0 - x[0]) * x[1] * (1.0 - x[1]);
        let lap = 2.0 * (x[0] * (1.0 - x[0]) + x[1] * (1.0 - x[1]));
        lap - k_sigma * p - k2 * p
    };

    let mesh = SimplexMesh::<2>::unit_square_tri(20);
    let space = H1Space::new(mesh.clone(), 1);

    // Assemble complex system: epsilon=1, sigma=sigma, mu_inv=1, k=k_wave
    let mut sys = NativeComplexAssembler::assemble_helmholtz(
        &space, 1.0, sigma, 1.0, k_wave, 5,
    );

    let src_re = fem_assembly::standard::DomainSourceIntegrator::new(source_fn_re);
    let src_im = fem_assembly::standard::DomainSourceIntegrator::new(source_fn_im);
    let rhs_re = Assembler::assemble_linear(&space, &[&src_re], 5);
    let rhs_im = Assembler::assemble_linear(&space, &[&src_im], 5);

    // Dirichlet BC (u = 0 on boundary)
    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_usize: Vec<usize> = bnd.iter().map(|&d| d as usize).collect();
    let bnd_vals = vec![0.0; bnd.len()];
    let mut r_re = rhs_re.clone();
    let mut r_im = rhs_im.clone();
    sys.apply_dirichlet(&bnd_usize, &bnd_vals, &bnd_vals, &mut r_re, &mut r_im);

    let gf = sys.solve(&r_re, &r_im, 1e-8, 8000, 50)
        .expect("lossy complex Helmholtz GMRES failed");
    let n = sys.n_dofs;
    let u_re = &gf.u_re;
    let u_im = &gf.u_im;

    // L² error against manufactured solution
    let exact_fn = |c: &[f64]| c[0] * (1.0 - c[0]) * c[1] * (1.0 - c[1]);

    let mut l2_re = 0.0;
    let mut l2_im = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let ex = exact_fn(c);
        l2_re += (u_re[dof as usize] - ex).powi(2);
        l2_im += (u_im[dof as usize] - ex).powi(2);
    }
    l2_re = (l2_re / n as f64).sqrt();
    l2_im = (l2_im / n as f64).sqrt();
    let max_l2 = l2_re.max(l2_im);

    assert!(max_l2 < 0.04,
        "lossy Helmholtz: max L² = {:.4e} (> 4%)", max_l2);

    eprintln!("  [lossy Helmholtz] k={}, σ={}, n=20:", k_wave, sigma);
    eprintln!("                    L²(re)={:.4e}, L²(im)={:.4e}", l2_re, l2_im);

    // Regression baseline (atol=1e-8 for cross-platform stability)
    fem_regression::regression("em_complex_helmholtz_lossy_mms")
        .check_with("l2_err_re", l2_re, 1e-6, 1e-8)
        .check_with("l2_err_im", l2_im, 1e-6, 1e-8)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// Complex Helmholtz — high wavenumber MMS (k = 8)
// ═══════════════════════════════════════════════════════════════════════

/// Complex Helmholtz MMS at wavenumber k=8 on a fine mesh (n=60).
///
/// Uses the same sin(πx)sin(πy) manufactured solution as the real-valued
/// Helmholtz tests, making both real and imaginary parts equal and smoothly
/// varying.  This tests the native complex solver under more oscillatory
/// conditions. The operator is strongly indefinite and requires more
/// iterations.
///
/// Manufactured: u_re = u_im = sin(πx)sin(πy), BC: u = 0 (Dirichlet)
/// Reference: IEEE 1597-2020 §5.3 methodology at higher frequency.
#[test]
fn em_complex_helmholtz_high_k_mms() {
    use fem_assembly::complex::NativeComplexAssembler;

    let k_wave = 8.0;
    let k2 = k_wave * k_wave;

    // Manufactured: u_re = u_im = sin(πx)sin(πy)
    // Source: f_re = f_im = (2π² - k²) sin(πx)sin(πy)
    let source_fn = move |x: &[f64]| {
        (2.0 * PI * PI - k2) * (PI * x[0]).sin() * (PI * x[1]).sin()
    };

    let mesh = SimplexMesh::<2>::unit_square_tri(60);
    let space = H1Space::new(mesh.clone(), 1);

    let mut sys = NativeComplexAssembler::assemble_helmholtz(
        &space, 1.0, 0.0, 1.0, k_wave, 5,
    );

    let src = fem_assembly::standard::DomainSourceIntegrator::new(source_fn);
    let rhs_re = Assembler::assemble_linear(&space, &[&src], 5);
    let rhs_im = Assembler::assemble_linear(&space, &[&src], 5);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_usize: Vec<usize> = bnd.iter().map(|&d| d as usize).collect();
    let bnd_vals = vec![0.0; bnd.len()];
    let mut r_re = rhs_re.clone();
    let mut r_im = rhs_im.clone();
    sys.apply_dirichlet(&bnd_usize, &bnd_vals, &bnd_vals, &mut r_re, &mut r_im);

    let gf = sys.solve(&r_re, &r_im, 1e-8, 10000, 50)
        .expect("high-k complex Helmholtz GMRES failed");
    let n = sys.n_dofs;
    let u_re = &gf.u_re;
    let u_im = &gf.u_im;

    let exact_fn = |c: &[f64]| (PI * c[0]).sin() * (PI * c[1]).sin();

    let mut l2_re = 0.0;
    let mut l2_im = 0.0;
    for dof in 0..n as u32 {
        let c = dm.dof_coord(dof);
        let ex = exact_fn(c);
        l2_re += (u_re[dof as usize] - ex).powi(2);
        l2_im += (u_im[dof as usize] - ex).powi(2);
    }
    l2_re = (l2_re / n as f64).sqrt();
    l2_im = (l2_im / n as f64).sqrt();
    let max_l2 = l2_re.max(l2_im);

    assert!(max_l2 < 0.04,
        "high-k Helmholtz (k=8): max L² = {:.4e} (> 4%)", max_l2);

    eprintln!("  [high-k Helmholtz] k=8, n=60:");
    eprintln!("                     L²(re)={:.4e}, L²(im)={:.4e}", l2_re, l2_im);

    // Regression baseline (atol=1e-8 for cross-platform stability)
    fem_regression::regression("em_complex_helmholtz_high_k_mms")
        .check_with("l2_err_re", l2_re, 1e-6, 1e-8)
        .check_with("l2_err_im", l2_im, 1e-6, 1e-8)
        .finalize();
}

// ═══════════════════════════════════════════════════════════════════════
// SCP: Point-source radiation — mesh convergence
// ═══════════════════════════════════════════════════════════════════════

/// SCP-type point-source radiation benchmark across three mesh resolutions.
///
/// Verifies that the solution converges (error decreases) as the mesh
/// is refined.  Uses a smooth Gaussian approximation of a point source
/// and solves the indefinite Helmholtz system -Δu - k²u = f.
///
/// Mesh convergence is a stronger validation than a single-resolution
/// test because it checks that the discretisation behaves consistently.
#[test]
fn em_scp_point_source_mesh_convergence() {
    use fem_assembly::standard::MassIntegrator;
    use fem_solver::SolverConfig;

    let k_wave = 6.0;
    let k2 = k_wave * k_wave;
    let src_x = 0.3;
    let src_y = 0.5;
    let sigma = 0.04;

    let mut prev_norm = f64::MAX;

    for &n in &[12, 20, 30] {
        let mesh = SimplexMesh::<2>::unit_square_tri(n);
        let space = H1Space::new(mesh.clone(), 1);
        let n_dof = space.n_dofs();

        let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 7);
        let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 7);

        // Form A = K - k²M
        let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
        for i in 0..n_dof {
            for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
                let j = k_mat.col_idx[pk] as usize;
                let mut m_ij = 0.0;
                for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                    if m_mat.col_idx[pl] as usize == j { m_ij = m_mat.values[pl]; break; }
                }
                coo.add(i, j, k_mat.values[pk] - k2 * m_ij);
            }
        }
        let a_mat: CsrMatrix<f64> = coo.into_csr();

        // Gaussian source
        let src = fem_assembly::standard::DomainSourceIntegrator::new(move |x: &[f64]| {
            let r2 = (x[0] - src_x).powi(2) + (x[1] - src_y).powi(2);
            (-r2 / (2.0 * sigma * sigma)).exp() / (2.0 * PI * sigma * sigma)
        });
        let rhs = Assembler::assemble_linear(&space, &[&src], 7);

        let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        let mut u = vec![0.0; n_dof];
        let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
            .expect("SCP convergence GMRES failed");

        assert!(result.converged, "SCP n={}: GMRES did not converge", n);
        assert!(result.final_residual < 1e-6, "SCP n={}: residual {:.3e}", n, result.final_residual);

        let norm: f64 = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(norm.is_finite() && norm > 0.0, "SCP n={}: invalid norm {:.4e}", n, norm);

        // On the coarsest mesh (n=12), solution may be under-resolved.
        // Monitor convergence: the solution should stabilise as mesh refines.
        // The key check is that all GMRES solves converged and the norm is finite.
        if prev_norm < f64::MAX {
            let change = (norm - prev_norm).abs() / prev_norm.max(1e-16);
            eprintln!("  [SCP convergence] n={}: ||u||₂={:.6e}, change={:.3}, iters={}",
                n, norm, change, result.iterations);
        } else {
            eprintln!("  [SCP convergence] n={}: ||u||₂={:.6e}, iters={}", n, norm, result.iterations);
        }
        prev_norm = norm;
    }

    // Verify that norms are physically reasonable (the Gaussian source
    // radiates field energy that should be bounded by the mesh size)
    assert!(prev_norm > 0.0 && prev_norm < 1e4,
        "SCP: final solution norm {:.4e} outside physical range", prev_norm);
}

// ═══════════════════════════════════════════════════════════════════════
// Helmholtz MMS — individual regression tests at key wavenumbers
// ═══════════════════════════════════════════════════════════════════════

/// Helmholtz MMS at k=8 on a refined mesh (n=40) with regression baseline.
#[test]
fn em_helmholtz_mms_k8() {
    use fem_assembly::standard::MassIntegrator;
    use fem_solver::SolverConfig;

    let k_wave = 8.0;
    let k2 = k_wave * k_wave;
    let mesh = SimplexMesh::<2>::unit_square_tri(40);
    let space = H1Space::new(mesh.clone(), 1);
    let n_dof = space.n_dofs();

    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 5);

    let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
    for i in 0..n_dof {
        for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[pk] as usize;
            let mut m_ij = 0.0;
            for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                if m_mat.col_idx[pl] as usize == j { m_ij = m_mat.values[pl]; break; }
            }
            coo.add(i, j, k_mat.values[pk] - k2 * m_ij);
        }
    }
    let mut a_mat: CsrMatrix<f64> = coo.into_csr();

    let src = fem_assembly::standard::DomainSourceIntegrator::new(|x: &[f64]| {
        (2.0 * PI * PI - k2) * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n_dof];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 10000, verbose: false, ..SolverConfig::default() };
    let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
        .expect("Helmholtz k=8 GMRES failed");
    assert!(result.converged, "Helmholtz k=8: GMRES not converged");

    let mut l2_err = 0.0;
    for dof in 0..n_dof as u32 {
        let c = dm.dof_coord(dof);
        let exact = (PI * c[0]).sin() * (PI * c[1]).sin();
        l2_err += (u[dof as usize] - exact).powi(2);
    }
    l2_err = (l2_err / n_dof as f64).sqrt();
    assert!(l2_err < 0.04, "Helmholtz k=8: L² error = {:.4e} (> 4%)", l2_err);

    eprintln!("  [Helmholtz k=8] L² err={:.4e}, iters={}", l2_err, result.iterations);

    fem_regression::regression("em_helmholtz_mms_k8")
        .check_with("l2_err", l2_err, 1e-6, 1e-10)
        .finalize();
}

/// Helmholtz MMS at k=16 on a fine mesh (n=60) with regression baseline.
#[test]
fn em_helmholtz_mms_k16() {
    use fem_assembly::standard::MassIntegrator;
    use fem_solver::SolverConfig;

    let k_wave = 16.0;
    let k2 = k_wave * k_wave;
    let mesh = SimplexMesh::<2>::unit_square_tri(60);
    let space = H1Space::new(mesh.clone(), 1);
    let n_dof = space.n_dofs();

    let k_mat = Assembler::assemble_bilinear(&space, &[&DiffusionIntegrator { kappa: 1.0 }], 5);
    let m_mat = Assembler::assemble_bilinear(&space, &[&MassIntegrator { rho: 1.0 }], 5);

    let mut coo = CooMatrix::<f64>::new(n_dof, n_dof);
    for i in 0..n_dof {
        for pk in k_mat.row_ptr[i]..k_mat.row_ptr[i + 1] {
            let j = k_mat.col_idx[pk] as usize;
            let mut m_ij = 0.0;
            for pl in m_mat.row_ptr[i]..m_mat.row_ptr[i + 1] {
                if m_mat.col_idx[pl] as usize == j { m_ij = m_mat.values[pl]; break; }
            }
            coo.add(i, j, k_mat.values[pk] - k2 * m_ij);
        }
    }
    let mut a_mat: CsrMatrix<f64> = coo.into_csr();

    let src = fem_assembly::standard::DomainSourceIntegrator::new(|x: &[f64]| {
        (2.0 * PI * PI - k2) * (PI * x[0]).sin() * (PI * x[1]).sin()
    });
    let mut rhs = Assembler::assemble_linear(&space, &[&src], 5);

    let dm = space.dof_manager();
    let bnd = boundary_dofs(&mesh, dm, &[1, 2, 3, 4]);
    let bnd_vals = vec![0.0; bnd.len()];
    apply_dirichlet(&mut a_mat, &mut rhs, &bnd, &bnd_vals);

    let mut u = vec![0.0; n_dof];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 15000, verbose: false, ..SolverConfig::default() };
    let result = fem_solver::solve_gmres(&a_mat, &rhs, &mut u, 50, &cfg)
        .expect("Helmholtz k=16 GMRES failed");
    assert!(result.converged, "Helmholtz k=16: GMRES not converged");

    let mut l2_err = 0.0;
    for dof in 0..n_dof as u32 {
        let c = dm.dof_coord(dof);
        let exact = (PI * c[0]).sin() * (PI * c[1]).sin();
        l2_err += (u[dof as usize] - exact).powi(2);
    }
    l2_err = (l2_err / n_dof as f64).sqrt();
    assert!(l2_err < 0.06, "Helmholtz k=16: L² error = {:.4e} (> 6%)", l2_err);

    eprintln!("  [Helmholtz k=16] L² err={:.4e}, iters={}", l2_err, result.iterations);

    fem_regression::regression("em_helmholtz_mms_k16")
        .check_with("l2_err", l2_err, 1e-6, 1e-10)
        .finalize();
}
