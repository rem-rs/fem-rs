//! # Example 53: 3-D Electrothermal Coupling on Tetrahedral Meshes
//!
//! Solves the steady electrothermal coupling problem on the unit cube [0,1]³
//! using P1 finite elements on an unstructured tetrahedral mesh.
//!
//! ## Physics
//!
//! Electric sub-problem (Laplace):
//! ```text
//!   -∇·(σ(T) ∇φ) = 0   in Ω = [0,1]³
//!   φ = 0               on z = 0  (face tag 1)
//!   φ = V               on z = 1  (face tag 2)
//!   ∂φ/∂n = 0           on all other faces (natural)
//! ```
//!
//! Thermal sub-problem (Poisson):
//! ```text
//!   -∇·(κ ∇T) = Q(x)   in Ω
//!   T = 0               on all faces (tags 1–6)
//! ```
//!
//! where the Joule heat source is `Q(x) = σ(T) |∇φ|²` (piecewise constant
//! per element), and the temperature-dependent conductivity is:
//!
//! ```text
//!   σ(T) = σ₀ · max(1 + σ_β · T_mean, 1e-12)
//! ```
//!
//! Coupling is resolved by fixed-point iteration with relaxation.
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex53_3d_electrothermal
//! cargo run --example mfem_ex53_3d_electrothermal -- --n 6 --voltage 2.0
//! ```
//!
//! ## Boundary tag convention for `unit_cube_tet`
//! | Tag | Face  | Outward normal |
//! |-----|-------|---------------|
//! |  1  | z = 0 | −z (bottom)  |
//! |  2  | z = 1 | +z (top)     |
//! |  3  | y = 0 | −y (front)   |
//! |  4  | y = 1 | +y (back)    |
//! |  5  | x = 0 | −x (left)    |
//! |  6  | x = 1 | +x (right)   |

use fem_assembly::{
    Assembler,
    postprocess::compute_element_gradients,
    standard::DiffusionIntegrator,
};
use fem_mesh::{SimplexMesh, topology::MeshTopology};
use fem_solver::{SolverConfig, solve_pcg_jacobi};
use fem_space::{
    H1Space,
    constraints::{apply_dirichlet, boundary_dofs},
    fe_space::FESpace,
};

// ─── Result type ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SolveResult3D {
    /// Number of degrees of freedom.
    n_dofs: usize,
    /// Number of tetrahedral elements.
    n_elems: usize,
    /// Whether the outer fixed-point iteration converged.
    converged: bool,
    /// Number of coupling iterations performed.
    iterations: usize,
    /// Final relative temperature change between iterations.
    final_relative_change: f64,
    /// Effective conductivity at convergence.
    sigma_effective: f64,
    /// L² norm of the electric potential φ.
    phi_norm: f64,
    /// L² norm of the temperature field T.
    temp_norm: f64,
    /// Integrated Joule power ∫ Q dΩ.
    joule_power: f64,
    /// Maximum temperature value.
    temp_max: f64,
}

// ─── Args ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Args {
    /// Mesh parameter: n×n×n cube cells (each split into 6 tets).
    n: usize,
    /// Applied voltage (φ on top face z=1).
    voltage: f64,
    /// Base electrical conductivity.
    sigma0: f64,
    /// Temperature coefficient: σ(T) = σ₀(1 + σ_β·T_mean).
    sigma_beta: f64,
    /// Thermal conductivity (isotropic).
    kappa: f64,
    /// Fixed-point relaxation factor.
    relax: f64,
    /// Maximum number of coupling iterations.
    max_coupling: usize,
    /// Convergence tolerance on relative temperature change.
    tol: f64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            n: 5,
            voltage: 1.0,
            sigma0: 1.0,
            sigma_beta: 0.0,
            kappa: 1.0,
            relax: 0.7,
            max_coupling: 30,
            tol: 1.0e-8,
        }
    }
}

// ─── Solver ──────────────────────────────────────────────────────────────────

fn solve_3d_electrothermal(args: &Args) -> SolveResult3D {
    let mesh = SimplexMesh::<3>::unit_cube_tet(args.n);
    let space = H1Space::new(mesh, 1);
    let n_dofs = space.n_dofs();
    let n_elems = space.mesh().n_elements();

    let cfg = SolverConfig {
        rtol: 1.0e-10,
        atol: 0.0,
        max_iter: 2000,
        verbose: false,
        ..SolverConfig::default()
    };

    // Dirichlet boundary sets:
    //   φ = 0 on face tag 1 (z=0),  φ = V on face tag 2 (z=1)
    //   T = 0 on all faces (tags 1–6)
    let bnd_bottom = boundary_dofs(space.mesh(), space.dof_manager(), &[1]);
    let bnd_top    = boundary_dofs(space.mesh(), space.dof_manager(), &[2]);
    let bnd_all    = boundary_dofs(space.mesh(), space.dof_manager(), &[1, 2, 3, 4, 5, 6]);

    let mut phi  = vec![0.0_f64; n_dofs];
    let mut temp = vec![0.0_f64; n_dofs];

    let mut sigma_eff = args.sigma0.max(1.0e-12);
    let mut final_rel = f64::INFINITY;
    let mut joule_power = 0.0_f64;
    let mut iters_done = 0usize;
    let mut converged  = false;

    for k in 0..args.max_coupling {
        // Update conductivity from mean temperature.
        let t_mean = temp.iter().sum::<f64>() / n_dofs as f64;
        sigma_eff = (args.sigma0 * (1.0 + args.sigma_beta * t_mean)).max(1.0e-12);

        // ── Electric solve: -div(σ∇φ) = 0 ──────────────────────────────────
        let sigma = sigma_eff;
        let mut a_phi = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa: sigma }],
            4, // quad order 4 is sufficient for P1 in 3D
        );
        let mut rhs_phi = vec![0.0_f64; n_dofs];
        apply_dirichlet(&mut a_phi, &mut rhs_phi, &bnd_bottom, &vec![0.0; bnd_bottom.len()]);
        apply_dirichlet(
            &mut a_phi,
            &mut rhs_phi,
            &bnd_top,
            &vec![args.voltage; bnd_top.len()],
        );
        solve_pcg_jacobi(&a_phi, &rhs_phi, &mut phi, &cfg).expect("electric solve failed");

        // ── Joule source Q = σ|∇φ|² (piecewise constant per element) ───────
        let grads = compute_element_gradients(&space, &phi);
        let q_elem: Vec<f64> = grads
            .iter()
            .map(|g| sigma * (g[0]*g[0] + g[1]*g[1] + g[2]*g[2]))
            .collect();
        joule_power = integrate_element_scalar_3d(space.mesh(), &q_elem);

        // ── Thermal solve: -div(κ∇T) = Q ───────────────────────────────────
        // Build the RHS directly from the element-wise Joule source.
        // For P1 tets: ∫_e N_i dx = vol_e / 4 for each corner node i.
        let mut rhs_t = vec![0.0_f64; n_dofs];
        for e in space.mesh().elem_iter() {
            let vol = tet_volume(space.mesh(), e);
            let q = q_elem[e as usize];
            let dofs = space.element_dofs(e);
            for &d in dofs.iter() {
                rhs_t[d as usize] += q * vol / 4.0;
            }
        }
        let kappa = args.kappa;
        let mut a_t = Assembler::assemble_bilinear(
            &space,
            &[&DiffusionIntegrator { kappa }],
            4,
        );
        apply_dirichlet(&mut a_t, &mut rhs_t, &bnd_all, &vec![0.0; bnd_all.len()]);

        let mut t_new = temp.clone();
        solve_pcg_jacobi(&a_t, &rhs_t, &mut t_new, &cfg).expect("thermal solve failed");

        // ── Relaxation & convergence check ───────────────────────────────────
        let mut diff2 = 0.0_f64;
        let mut base2 = 0.0_f64;
        for i in 0..n_dofs {
            let relaxed = (1.0 - args.relax) * temp[i] + args.relax * t_new[i];
            let d = relaxed - temp[i];
            diff2 += d * d;
            base2 += relaxed * relaxed;
            temp[i] = relaxed;
        }
        final_rel = diff2.sqrt() / base2.sqrt().max(1.0e-14);
        iters_done = k + 1;
        if final_rel <= args.tol {
            converged = true;
            break;
        }
    }

    let phi_norm  = l2_norm(&phi);
    let temp_norm = l2_norm(&temp);
    let temp_max  = temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    SolveResult3D {
        n_dofs,
        n_elems,
        converged,
        iterations: iters_done,
        final_relative_change: final_rel,
        sigma_effective: sigma_eff,
        phi_norm,
        temp_norm,
        joule_power,
        temp_max,
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Integrate a piecewise-constant element field over a 3-D tet mesh.
/// Volume of a tet with Jacobian J is |det(J)| / 6.
fn integrate_element_scalar_3d(mesh: &SimplexMesh<3>, elem_values: &[f64]) -> f64 {
    let mut acc = 0.0_f64;
    for (e, &val) in mesh.elem_iter().zip(elem_values.iter()) {
        let vol = tet_volume(mesh, e);
        acc += val * vol;
    }
    acc
}

fn tet_volume(mesh: &SimplexMesh<3>, elem: u32) -> f64 {
    let ns = mesh.elem_nodes(elem);
    let a = mesh.coords_of(ns[0]);
    let b = mesh.coords_of(ns[1]);
    let c = mesh.coords_of(ns[2]);
    let d = mesh.coords_of(ns[3]);
    // Edges from a:
    let ab = [b[0]-a[0], b[1]-a[1], b[2]-a[2]];
    let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
    let ad = [d[0]-a[0], d[1]-a[1], d[2]-a[2]];
    // Triple product ab · (ac × ad)
    let cross = [
        ac[1]*ad[2] - ac[2]*ad[1],
        ac[2]*ad[0] - ac[0]*ad[2],
        ac[0]*ad[1] - ac[1]*ad[0],
    ];
    let triple = ab[0]*cross[0] + ab[1]*cross[1] + ab[2]*cross[2];
    triple.abs() / 6.0
}

fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args();
    println!("=== fem-rs Example 53: 3-D Electrothermal Coupling ===");
    println!(
        "  n={}, V={}, σ₀={}, σ_β={}, κ={}, max_coupling={}, tol={:.1e}",
        args.n, args.voltage, args.sigma0, args.sigma_beta, args.kappa,
        args.max_coupling, args.tol
    );

    let r = solve_3d_electrothermal(&args);

    println!("  n_dofs = {}", r.n_dofs);
    println!("  n_elems = {}", r.n_elems);
    println!("  converged = {} ({} iters, Δrel={:.3e})", r.converged, r.iterations, r.final_relative_change);
    println!("  σ_eff = {:.6e}", r.sigma_effective);
    println!("  ||φ||₂ = {:.6e}", r.phi_norm);
    println!("  ||T||₂ = {:.6e}", r.temp_norm);
    println!("  T_max  = {:.6e}", r.temp_max);
    println!("  Joule  = {:.6e}", r.joule_power);
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--n"          => { i += 1; a.n          = argv[i].parse().unwrap(); }
            "--voltage"    => { i += 1; a.voltage    = argv[i].parse().unwrap(); }
            "--sigma0"     => { i += 1; a.sigma0     = argv[i].parse().unwrap(); }
            "--sigma_beta" => { i += 1; a.sigma_beta = argv[i].parse().unwrap(); }
            "--kappa"      => { i += 1; a.kappa      = argv[i].parse().unwrap(); }
            "--relax"      => { i += 1; a.relax      = argv[i].parse().unwrap(); }
            "--max_coupling" => { i += 1; a.max_coupling = argv[i].parse().unwrap(); }
            "--tol"        => { i += 1; a.tol        = argv[i].parse().unwrap(); }
            _ => {}
        }
        i += 1;
    }
    a
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            n: 4,
            voltage: 1.0,
            sigma0: 1.0,
            sigma_beta: 0.0,
            kappa: 1.0,
            relax: 0.7,
            max_coupling: 30,
            tol: 1.0e-8,
        }
    }

    /// Smoke test: 3-D solve produces finite, positive results and converges.
    #[test]
    fn ex53_3d_electrothermal_converges_and_is_finite() {
        let r = solve_3d_electrothermal(&base_args());
        assert!(r.converged, "coupling iteration should converge");
        assert!(r.phi_norm > 0.0, "phi should be nonzero");
        assert!(r.temp_norm > 0.0, "temperature should be nonzero");
        assert!(r.joule_power > 0.0, "Joule power should be positive");
        assert!(r.phi_norm.is_finite());
        assert!(r.temp_norm.is_finite());
        assert!(r.joule_power.is_finite());
    }

    /// DOF count should match (n+1)³ nodes for P1 on a structured tet mesh.
    #[test]
    fn ex53_dof_count_matches_p1_nodes() {
        let r = solve_3d_electrothermal(&base_args()); // n=4
        let expected = (4 + 1_usize).pow(3); // 125 nodes
        assert_eq!(r.n_dofs, expected, "P1 DOFs should equal (n+1)³ = {expected}");
    }

    /// Higher applied voltage → more Joule power (quadratic scaling: P ~ V²).
    #[test]
    fn ex53_joule_power_scales_quadratically_with_voltage() {
        let mut a1 = base_args(); a1.voltage = 1.0;
        let mut a2 = base_args(); a2.voltage = 2.0;
        let r1 = solve_3d_electrothermal(&a1);
        let r2 = solve_3d_electrothermal(&a2);
        // P ~ V², so P(2V)/P(V) ≈ 4.
        let ratio = r2.joule_power / r1.joule_power;
        assert!(
            (ratio - 4.0).abs() < 0.05,
            "Joule power should scale as V²: got ratio {ratio:.4}, expected ~4.0"
        );
    }

    /// Higher thermal conductivity κ → lower peak temperature (same source, more cooling).
    #[test]
    fn ex53_higher_kappa_gives_lower_temperature() {
        let mut low_k  = base_args(); low_k.kappa  = 0.5;
        let mut high_k = base_args(); high_k.kappa = 5.0;
        let r_low  = solve_3d_electrothermal(&low_k);
        let r_high = solve_3d_electrothermal(&high_k);
        assert!(
            r_high.temp_max < r_low.temp_max,
            "higher κ should lower peak T: κ=0.5→{:.4e}, κ=5.0→{:.4e}",
            r_low.temp_max, r_high.temp_max
        );
    }

    /// Near-zero conductivity → negligible Joule power.
    #[test]
    fn ex53_near_zero_sigma_gives_negligible_joule_power() {
        let mut a = base_args();
        a.sigma0 = 1.0e-8;
        a.sigma_beta = 0.0;
        let r = solve_3d_electrothermal(&a);
        assert!(
            r.joule_power < 1.0e-6,
            "near-zero σ should give negligible Joule power: {:.4e}", r.joule_power
        );
    }

    /// Positive σ_β (self-heating increase) should yield more Joule power
    /// than σ_β = 0 for the same applied voltage.
    #[test]
    fn ex53_positive_sigma_beta_increases_joule_power() {
        let mut zero_fb = base_args(); zero_fb.sigma_beta = 0.0;
        let mut pos_fb  = base_args(); pos_fb.sigma_beta  = 0.5;
        let r_zero = solve_3d_electrothermal(&zero_fb);
        let r_pos  = solve_3d_electrothermal(&pos_fb);
        assert!(
            r_pos.joule_power >= r_zero.joule_power,
            "positive σ_β should not decrease Joule power: zero={:.4e} pos={:.4e}",
            r_zero.joule_power, r_pos.joule_power
        );
    }

    /// 3-D mesh refinement: n=4 → n=6 should increase DOF count and element count.
    #[test]
    fn ex53_mesh_refinement_increases_dof_and_elem_count() {
        let coarse = solve_3d_electrothermal(&base_args()); // n=4
        let mut fine_args = base_args(); fine_args.n = 6;
        let fine = solve_3d_electrothermal(&fine_args);
        assert!(fine.n_dofs > coarse.n_dofs, "refinement should add DOFs");
        assert!(fine.n_elems > coarse.n_elems, "refinement should add elements");
    }

    /// Electric potential norm should scale linearly with applied voltage.
    #[test]
    fn ex53_phi_norm_scales_linearly_with_voltage() {
        let mut a1 = base_args(); a1.voltage = 0.5;
        let mut a2 = base_args(); a2.voltage = 1.5;
        let r1 = solve_3d_electrothermal(&a1);
        let r2 = solve_3d_electrothermal(&a2);
        let ratio = r2.phi_norm / r1.phi_norm;
        let expected = 1.5 / 0.5; // = 3.0
        assert!(
            (ratio - expected).abs() < 0.01,
            "||φ||₂ should scale linearly with voltage: got {ratio:.4}, expected ~{expected:.4}"
        );
    }
}
