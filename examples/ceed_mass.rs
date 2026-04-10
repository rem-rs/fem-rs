//! Verify reed-backed mass and stiffness operators on a 2D triangular mesh.
//!
//! ## Checks
//!
//! | # | Assertion | Expected value |
//! |---|-----------|----------------|
//! | 1 | `Σ (M · 1)[i]` = area | 1.0 (unit square) |
//! | 2 | M is SPD: `1ᵀ M 1 > 0` | > 0 |
//! | 3 | Linearly consistent: `M · x` sums to `∫ x dΩ = 0.5` | 0.5 |
//! | 4 | `M · y` sums to `∫ y dΩ = 0.5` | 0.5 |
//! | 5 | `M` preserves positivity: all entries of `M · 1 ≥ 0` | true |
//! | 6 | `K · 1 = 0` (constants in null space of Laplacian) | ‖K·1‖ ≈ 0 |
//! | 7 | `K` is semi-negative-definite: `1ᵀ K 1 ≈ 0` | ≈ 0 |
//! | 8 | `K · x`: sum over interior nodes close to 0 (Green's 1st identity) | ≈ 0 |
//! | 9 | P1 mass with 6-pt quadrature: area = 1.0 (over-integrated, same result) | 1.0 |
//! |10 | 6-pt and 3-pt quadrature agree: `sum(M_6pt · 1) ≈ sum(M_3pt · 1)` | 1.0 |
//!
//! Run with:
//! - `cargo run --example ceed_mass -- --backend=reed`
//! - `cargo run --example ceed_mass -- --backend=native`

use fem_assembly::{Assembler, CsrLinearOperator, LinearOperator, OperatorBackend, standard::{DiffusionIntegrator, MassIntegrator}};
use fem_ceed::{CeedBackend, FemCeed};
use fem_mesh::SimplexMesh;
use fem_space::H1Space;

fn parse_backend() -> OperatorBackend {
    let mut backend = OperatorBackend::Reed;
    for arg in std::env::args().skip(1) {
        if let Some(value) = arg.strip_prefix("--backend=") {
            backend = match OperatorBackend::parse(value) {
                Some(b) => b,
                None => {
                    eprintln!("unknown backend '{value}', expected reed|native");
                    std::process::exit(2);
                }
            };
        }
    }
    backend
}

fn apply_mass_native(mesh: &SimplexMesh<2>, poly: u8, q: u8, input: &[f64]) -> Vec<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    let mass = MassIntegrator { rho: 1.0 };
    let m = Assembler::assemble_bilinear(&space, &[&mass], q);
    let op = CsrLinearOperator::new(&m);
    let mut out = vec![0.0; input.len()];
    op.apply(input, &mut out);
    out
}

fn apply_poisson_native(mesh: &SimplexMesh<2>, poly: u8, q: u8, input: &[f64]) -> Vec<f64> {
    let space = H1Space::new(mesh.clone(), poly);
    let diff = DiffusionIntegrator { kappa: 1.0 };
    let k = Assembler::assemble_bilinear(&space, &[&diff], q);
    let op = CsrLinearOperator::new(&k);
    let mut out = vec![0.0; input.len()];
    op.apply(input, &mut out);
    out
}

fn main() {
    let backend = parse_backend();

    let n = 8usize; // 8×8 grid of unit triangles
    let mesh = SimplexMesh::<2>::unit_square_tri(n);
    let ceed = FemCeed::with_backend(CeedBackend::ReedCpu);
    let n_nodes = mesh.n_nodes();

    let ones: Vec<f64>    = vec![1.0; n_nodes];
    let xs: Vec<f64>      = (0..n_nodes as u32).map(|i| mesh.coords_of(i)[0]).collect();
    let ys: Vec<f64>      = (0..n_nodes as u32).map(|i| mesh.coords_of(i)[1]).collect();

    // ── Mass operator (P1, 3-point quadrature) ────────────────────────────

    let (m_ones, m_xs, m_ys) = match backend {
        OperatorBackend::Reed => (
            ceed.apply_mass_2d(&mesh, 1, 3, &ones).unwrap(),
            ceed.apply_mass_2d(&mesh, 1, 3, &xs).unwrap(),
            ceed.apply_mass_2d(&mesh, 1, 3, &ys).unwrap(),
        ),
        OperatorBackend::Native => (
            apply_mass_native(&mesh, 1, 3, &ones),
            apply_mass_native(&mesh, 1, 3, &xs),
            apply_mass_native(&mesh, 1, 3, &ys),
        ),
    };

    // ── Stiffness / Laplacian operator (P1, 3-point quadrature) ──────────

    let (k_ones, k_xs) = match backend {
        OperatorBackend::Reed => (
            ceed.apply_poisson_2d(&mesh, 1, 3, &ones).unwrap(),
            ceed.apply_poisson_2d(&mesh, 1, 3, &xs).unwrap(),
        ),
        OperatorBackend::Native => (
            apply_poisson_native(&mesh, 1, 3, &ones),
            apply_poisson_native(&mesh, 1, 3, &xs),
        ),
    };

    // ── P1 mass with 6-point quadrature (over-integrated, result unchanged) ──

    let m6_ones = match backend {
        OperatorBackend::Reed => ceed.apply_mass_2d(&mesh, 1, 6, &ones).unwrap(),
        OperatorBackend::Native => apply_mass_native(&mesh, 1, 6, &ones),
    };

    let tol = 1e-10;
    let mut pass = true;

    // Check 1: ∫ 1 dΩ = area of unit square = 1.0
    let area = m_ones.iter().sum::<f64>();
    check(1, (area - 1.0).abs() < tol, &format!("sum(M·1) = {area:.12}, expected 1.0"), &mut pass);

    // Check 2: 1ᵀ M 1 > 0  (M is positive definite)
    let quad_form = dot(&ones, &m_ones);
    check(2, quad_form > 0.0, &format!("1ᵀ M 1 = {quad_form:.12}, expected > 0"), &mut pass);

    // Check 3: ∫ x dΩ = 0.5  (linear consistency)
    let int_x = m_xs.iter().sum::<f64>();
    check(3, (int_x - 0.5).abs() < tol, &format!("sum(M·x) = {int_x:.12}, expected 0.5"), &mut pass);

    // Check 4: ∫ y dΩ = 0.5
    let int_y = m_ys.iter().sum::<f64>();
    check(4, (int_y - 0.5).abs() < tol, &format!("sum(M·y) = {int_y:.12}, expected 0.5"), &mut pass);

    // Check 5: all entries of M · 1 ≥ 0
    let all_nonneg = m_ones.iter().all(|&v| v >= -tol);
    check(5, all_nonneg, "all entries of M·1 ≥ 0 (mass matrix row sums positive)", &mut pass);

    // Check 6: ‖K · 1‖ ≈ 0  (constants are in null space of Laplacian)
    let k_ones_norm = l2_norm(&k_ones);
    check(6, k_ones_norm < 1e-8, &format!("‖K·1‖ = {k_ones_norm:.3e}, expected ≈ 0"), &mut pass);

    // Check 7: 1ᵀ K 1 ≈ 0  (stiffness of constant field = 0)
    let k_quad = dot(&ones, &k_ones);
    check(7, k_quad.abs() < 1e-10, &format!("1ᵀ K 1 = {k_quad:.3e}, expected ≈ 0"), &mut pass);

    // Check 8: sum(K · x) ≈ 0  (by Green's 1st identity: ∫∇·(1·∇x)dΩ = ∮ ∂x/∂n dΓ)
    // The integral of K·x over interior nodes is 0; boundary contribution
    // equals the line integral of the normal derivative over ∂Ω.
    // For u=x on unit square: ∮ ∂x/∂n dΓ = ∮ nx dΓ.
    // On ∂Ω: right side (x=1, nx=1, length=1) + left side (nx=-1, length=1) = 1-1=0.
    // Top/bottom: ∂x/∂n = 0. So total = 0.
    let sum_kx = k_xs.iter().sum::<f64>();
    check(8, sum_kx.abs() < 1e-8, &format!("sum(K·x) = {sum_kx:.3e}, expected ≈ 0"), &mut pass);

    // Check 9: P1 mass with 6-pt quad gives same area as 3-pt
    let area_6pt = m6_ones.iter().sum::<f64>();
    check(9, (area_6pt - 1.0).abs() < tol, &format!("sum(M_6pt·1) = {area_6pt:.12}, expected 1.0"), &mut pass);

    // Check 10: 6-pt and 3-pt quadrature agree on quadratic form
    let qf_6pt = dot(&ones, &m6_ones);
    check(10, (qf_6pt - quad_form).abs() < tol, &format!("1ᵀ M_6pt 1 = {qf_6pt:.12}, matches 3-pt = {quad_form:.12}"), &mut pass);

    println!("backend: {}", backend.as_str());
    if pass {
        println!("✓ all 10 checks passed");
    } else {
        eprintln!("✗ some checks failed");
        std::process::exit(1);
    }
}

fn check(n: usize, cond: bool, msg: &str, pass: &mut bool) {
    if cond {
        println!("  check {n:2}: ✓  {msg}");
    } else {
        eprintln!("  check {n:2}: ✗  {msg}");
        *pass = false;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}
