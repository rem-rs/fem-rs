//! # Example 33 — Fractional Diffusion  [1:1 translation of MFEM ex33 + ex33.hpp]
//!
//! Solves the fractional diffusion equation
//!
//! ```text
//!   (-Δ)^α u = f  in Ω,     u = 0  on ∂Ω,     0 < α
//! ```
//!
//! The integer part is handled by solving `(-Δ)^N g = f` (N = floor(α)) with
//! H¹ + Diffusion/Mass integrators and PCG-GSSmoother (the ex27–ex29 1:1
//! infrastructure).  The fractional remainder `(-Δ)^{α-N}` is approximated by
//! a rational (partial-fraction) expansion generated with the triple-A (AAA)
//! algorithm [1] of Nakatsukasa & Trefethen, as implemented in MFEM's
//! `examples/ex33.hpp`:
//!
//! ```text
//!   A^{-α+N} ≈ Σ_{i=0}^M c_i (A + d_i M)^{-1}
//! ```
//!
//! We solve the M+1 independent shifted systems `(A + d_i M) u_i = c_i g`
//! and sum the solutions.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex33_fractional_diffusion
//! cargo run --example mfem_ex33_fractional_diffusion -- -m data/star.mesh -alpha 0.33 -o 2
//! cargo run --example mfem_ex33_fractional_diffusion -- -m data/inline-quad.mesh -ver -alpha 1.2 -o 2 -r 2
//! ```
//!
//! ## References
//! [1] Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). The AAA algorithm
//!     for rational approximation. SIAM J. Sci. Comput. 40(3), A1494-A1522.
//! [2] Harizanov, S., et al. (2020). Analysis of numerical methods for spectral
//!     fractional elliptic equations… J. Comput. Phys. 408, 109285.

use std::collections::HashSet;
use std::f64::consts::PI;

use fem_assembly::{
    standard::{DiffusionIntegrator, DomainSourceIntegrator, MassIntegrator},
    Assembler, GridFunction,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_pcg_gssmoother, fmt_g, PrintLevel, SolverConfig};
use fem_space::{
    constraints::{boundary_dofs, form_linear_system},
    fe_space::FESpace,
    H1Space,
};
#[cfg(test)]
use fem_examples::rational_approximation::{
    compute_partial_fraction_approximation, rational_approximation_aaa, weighted_poly_product,
};

/// MFEM ex33.hpp `ComputePartialFractionApproximation`, no-LAPACK branch
/// (this is the configuration of the pinned C++ reference build): prints the
/// banner and returns the hard-coded partial-fraction tables for
/// `alpha ∈ {0.33, 0.5, 0.99}`; any other exponent silently becomes 0.5
/// *inside the function* (ex33.cpp restores its own exponent copy
/// afterwards, so only the "=> Using precomputed values" line sees it).
fn precomputed_partial_fraction_approximation(alpha: f64) -> (Vec<f64>, Vec<f64>) {
    println!();
    println!("{}", "=".repeat(80));
    println!("MFEM is compiled without LAPACK.");
    println!("Using precomputed values for PartialFractionApproximation.");
    println!("Only alpha = 0.33, 0.5, and 0.99 are available.");
    println!("The default is alpha = 0.5.");
    println!("{}", "=".repeat(80));
    println!();

    const EPS: f64 = f64::EPSILON;
    let (coeffs, poles) = if (alpha - 0.33).abs() < EPS {
        (
            vec![
                1.821898e+03, 9.101221e+01, 2.650611e+01,
                1.174937e+01, 6.140444e+00, 3.441713e+00,
                1.985735e+00, 1.162634e+00, 6.891560e-01,
                4.111574e-01, 2.298736e-01,
            ],
            vec![
                -4.155583e+04, -2.956285e+03, -8.331715e+02,
                -3.139332e+02, -1.303448e+02, -5.563385e+01,
                -2.356255e+01, -9.595516e+00, -3.552160e+00,
                -1.032136e+00, -1.241480e-01,
            ],
        )
    } else if (alpha - 0.99).abs() < EPS {
        (
            vec![
                2.919591e-02, 1.419750e-02, 1.065798e-02,
                9.395094e-03, 8.915329e-03, 8.822991e-03,
                9.058247e-03, 9.814521e-03, 1.180396e-02,
                1.834554e-02, 9.840482e-01,
            ],
            vec![
                -1.069683e+04, -1.769370e+03, -5.718374e+02,
                -2.242095e+02, -9.419132e+01, -4.031012e+01,
                -1.701525e+01, -6.810088e+00, -2.382810e+00,
                -5.700059e-01, -1.384324e-03,
            ],
        )
    } else {
        // Default branch: the exponent is redefined to 0.5 within the
        // function for the banner line below.
        (
            vec![
                2.290262e+02, 2.641819e+01, 1.005566e+01,
                5.390411e+00, 3.340725e+00, 2.211205e+00,
                1.508883e+00, 1.049474e+00, 7.462709e-01,
                5.482686e-01, 4.232510e-01, 3.578967e-01,
            ],
            vec![
                -3.168211e+04, -3.236077e+03, -9.868287e+02,
                -3.945597e+02, -1.738889e+02, -7.925178e+01,
                -3.624992e+01, -1.629196e+01, -6.982956e+00,
                -2.679984e+00, -7.782607e-01, -7.649166e-02,
            ],
        )
    };

    println!(
        "=> Using precomputed values for alpha = {}",
        fmt_g(if (alpha - 0.33).abs() < EPS || (alpha - 0.99).abs() < EPS {
            alpha
        } else {
            0.5
        })
    );
    println!();
    (coeffs, poles)
}
fn main() {
    let args = parse_args();
    // C++ ex33 echoes the parsed options (`args.PrintOptions(cout)`,
    // ex33.cpp:128) before any other output; the echo mirrors
    // OptionsParser::PrintOptions byte-for-byte (ENABLE pair prints the
    // long_name whose value is true).
    println!("Options used:");
    println!("   --mesh {}", args.mesh);
    println!("   --order {}", args.order);
    println!("   --refs {}", args.refs);
    println!("   --alpha {}", args.alpha);
    println!("   {}", if args.visualization { "--visualization" } else { "--no-visualization" });
    println!("   {}", if args.verification { "--verification" } else { "--no-verification" });

    // ── 2. Compute the rational expansion coefficients (ex33.hpp) ──────────
    let power_of_laplace = args.alpha.floor() as i32;
    let exponent_to_approximate = args.alpha - power_of_laplace as f64;
    let integer_order = exponent_to_approximate.abs() <= 1e-12;

    // (coeffs[i], poles[i]) with d_i = -poles[i] > 0.
    let (coeffs, poles) = if !integer_order {
        println!(
            "Approximating the fractional exponent {}",
            fmt_g(exponent_to_approximate)
        );
        // MFEM 4.10 serial builds carry no LAPACK: ex33.hpp's
        // ComputePartialFractionApproximation then prints the no-LAPACK banner
        // and falls back to the hard-coded tables for alpha ∈ {0.33, 0.5,
        // 0.99} (anything else silently becomes 0.5 inside the function).
        precomputed_partial_fraction_approximation(exponent_to_approximate)
    } else {
        println!("Treating integer order PDE.");
        (Vec::new(), Vec::new())
    };

    // ── 3. Read the mesh ────────────────────────────────────────────────────
    let mfem = read_mfem_file(&args.mesh).expect("failed to read MFEM mesh file");
    let mesh0: Mesh<2> = mfem.mesh2d.expect("ex33 expects a 2D mesh");
    let dim = 2usize;

    // ── 4. Uniform refinement ───────────────────────────────────────────────
    let mut mesh = mesh0;
    for _ in 0..args.refs {
        mesh = refine_uniform(&mesh);
    }

    // ── 5. H¹ finite element space ──────────────────────────────────────────
    let space = H1Space::new(mesh.clone(), args.order as u8);
    let n_dofs = space.n_dofs();
    println!("Number of degrees of freedom: {}", n_dofs);

    // ── 6. Essential (Dirichlet) boundary DOFs — all boundary attributes ────
    let dm = space.dof_manager();
    let all_tags = mesh.unique_boundary_tags();
    let ess_bdr = if !all_tags.is_empty() {
        boundary_dofs(&mesh, dm, &all_tags)
    } else {
        Vec::new()
    };
    let ess_vals = vec![0.0_f64; ess_bdr.len()];
    let ess_set: HashSet<usize> = ess_bdr.iter().map(|&d| d as usize).collect();

    // ── 7-9. Load f and linear form b(.) ────────────────────────────────────
    // Verification: f(x) = (dim·π²)^α · ∏_i sin(π x_i)  ⇒  (-Δ)^α u = f with
    // u = ∏ sin(π x_i).  Otherwise f = 1 (matching C++ ex33).
    let source = DomainSourceIntegrator::new(move |x: &[f64]| -> f64 {
        if args.verification {
            let mut val = 1.0;
            for &xi in x {
                val *= (PI * xi).sin();
            }
            (x.len() as f64 * PI * PI).powf(args.alpha) * val
        } else {
            1.0
        }
    });
    let q_int = (2 * args.order) as u8; // MFEM integrator order = 2p
    let mut b = Assembler::assemble_linear(&space, &[&source], q_int);

    let cfg = SolverConfig {
        // MFEM's free `PCG(op, prec, b, x, print, maxit, RTOL, ATOL)` wraps a
        // CGSolver with SetRelTol(sqrt(RTOL)) / SetAbsTol(sqrt(ATOL)); the CG
        // stopping rule is `nom <= max(nom0·rel_tol², abs_tol²)`.  ex33 calls
        // PCG(..., 1e-12, 0.0) → rel_tol = sqrt(1e-12) here.
        rtol: 1e-6,
        atol: 0.0,
        max_iter: 300,
        verbose: true,
        print_level: PrintLevel::FirstAndLast,
    };

    let mut u = vec![0.0_f64; n_dofs];

    // ── 10. Integer-order part: solve (-Δ)^N g = f ──────────────────────────
    if power_of_laplace > 0 {
        // 10.1-10.2 Stiffness and mass matrices.
        let diff = DiffusionIntegrator { kappa: 1.0 };
        let mass_integ = MassIntegrator { rho: 1.0 };
        let k_mat = Assembler::assemble_bilinear(&space, &[&diff], q_int);
        let mass = Assembler::assemble_bilinear(&space, &[&mass_integ], q_int);

        // 10.3 Form the linear system (once; Op/B/X reused in the loop).
        let mut g_vec = vec![0.0_f64; n_dofs]; // GridFunction g (initial 0)
        let mut mat = k_mat.clone();
        let mut x = g_vec.clone();
        let mut B = b.clone();
        form_linear_system(&mut mat, &mut B, &mut x, &ess_bdr, &ess_vals);

        println!("\nComputing (-Δ) ^ -{} ( f ) ", power_of_laplace);
        for i in 0..power_of_laplace {
            // 10.4 Solve Op X = B (N times).
            solve_pcg_gssmoother(&mat, &B, &mut x, &cfg).expect("PCG failed");

            // 10.5 Recover the solution g in the last step.
            if i == power_of_laplace - 1 {
                g_vec = x.clone();
                if integer_order && args.verification {
                    for j in 0..n_dofs {
                        u[j] += g_vec[j];
                    }
                }
            }

            // 10.6 Prepare for next iteration: B = M·X; X[free] = 0.
            mass.spmv(&x, &mut B);
            for j in 0..n_dofs {
                if !ess_set.contains(&j) {
                    x[j] = 0.0;
                }
            }
        }

        // 10.7 b now carries B = M·X (the mass-scaled right-hand side for the
        //     next integer step / the fractional part).  Mirrors ex33.cpp
        //     (no restriction matrix in serial): b = B.
        b = B;
    }

    // ── 11. Fractional part: Σ_i c_i (A + d_i M)^{-1} g ─────────────────────
    if !integer_order {
        for i in 0..coeffs.len() {
            println!(
                "\nSolving PDE -Δ u + {} u = {} g ",
                fmt_g(-poles[i]),
                fmt_g(coeffs[i])
            );

            // 11.2 a(.,.) = Diffusion + d_i·Mass with d_i = -poles[i].
            let diff = DiffusionIntegrator { kappa: 1.0 };
            let mass_d = MassIntegrator { rho: -poles[i] };
            let integs: Vec<&dyn fem_assembly::BilinearIntegrator> = vec![&diff, &mass_d];
            let a_mat = Assembler::assemble_bilinear(&space, &integs, q_int);

            // 11.3 Form the linear system (x = 0 initial).
            let mut mat = a_mat;
            let mut x = vec![0.0_f64; n_dofs];
            let mut B = b.clone();
            form_linear_system(&mut mat, &mut B, &mut x, &ess_bdr, &ess_vals);

            // 11.4 Solve A X = B.
            solve_pcg_gssmoother(&mat, &B, &mut x, &cfg).expect("PCG failed");

            // 11.6 Accumulate: u += coeffs[i]·x.
            for j in 0..n_dofs {
                u[j] += coeffs[i] * x[j];
            }
        }
    }

    // ── 12. (optional) Verify the solution ──────────────────────────────────
    if args.verification {
        let solution = |x: &[f64]| -> f64 {
            let mut val = 1.0;
            for &xi in x {
                val *= (PI * xi).sin();
            }
            val
        };
        let gf = GridFunction::new(&space, u.clone());
        // MFEM ComputeL2Error default intorder = 2*order + 3.
        let l2_error = gf.compute_l2_error(&solution, (2 * args.order + 3) as u8);

        let (manufactured_solution, expected_mesh) = match dim {
            1 => ("sin(π x)", "inline_segment.mesh"),
            2 => ("sin(π x) sin(π y)", "inline_quad.mesh"),
            _ => ("sin(π x) sin(π y) sin(π z)", "inline_hex.mesh"),
        };

        println!("\n{}", "=".repeat(80));
        println!("\nSolution Verification in {}D \n", dim);
        println!("Manufactured solution : {}", manufactured_solution);
        println!("Expected mesh         : {}", expected_mesh);
        println!("Your mesh             : {}", args.mesh);
        println!("L2 error              : {}", fmt_g(l2_error));
        println!("\n{}", "=".repeat(80));
    }
}


// ═══════════════════════════════════════════════════════════════════════════
//  CLI
// ═══════════════════════════════════════════════════════════════════════════

struct Args {
    mesh: String,
    order: usize,
    refs: usize,
    alpha: f64,
    visualization: bool,
    verification: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: "data/star.mesh".to_string(),
        order: 2,
        refs: 3,
        alpha: 0.33,
        visualization: false,
        verification: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => {
                a.mesh = it.next().unwrap_or_else(|| a.mesh.clone());
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.order);
            }
            "-r" | "--refs" => {
                a.refs = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.refs);
            }
            "-alpha" | "--alpha" => {
                a.alpha = it.next().and_then(|s| s.parse().ok()).unwrap_or(a.alpha);
            }
            "-vis" | "--visualization" => a.visualization = true,
            "-no-vis" | "--no-visualization" => a.visualization = false,
            "-ver" | "--verification" => a.verification = true,
            "-no-ver" | "--no-verification" => a.verification = false,
            _ => {}
        }
    }
    a
}

// ═══════════════════════════════════════════════════════════════════════════
//  Tests — AAA coefficients vs the C++ (LAPACK) reference dumps
//  (tools/ex33_cpp_helper/dump_alpha0{33,20}.txt)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_diff(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
    }

    #[test]
    fn aaa_alpha033_matches_cpp_reference() {
        // C++ (dggev/LAPACK) reference for alpha = 0.33:
        // 12 support points, 11 poles, 11 coeffs; zeros 11 → 10 after
        // DeleteFirst(0.0).
        let (coeffs, poles) = compute_partial_fraction_approximation(0.33);
        assert_eq!(poles.len(), 11, "poles count");
        assert_eq!(coeffs.len(), 11, "coeffs count");

        // Reference values (C++ dump): c[0]=1821.897761064293 d[0]=41555.825674422296 …
        let ref_c = [
            1821.897761064293,
            91.012214185090741,
            26.506115002118285,
            11.749374295756416,
            6.1404438421287368,
            3.4417133750353388,
            1.9857347681613946,
            1.1626337734775813,
            0.6891560133453889,
            0.41115744945694915,
            0.22987359842320917,
        ];
        let ref_d = [
            41555.825674422296,
            2956.2854312402396,
            833.1714772646817,
            313.93322142885148,
            130.34483906605598,
            55.633854482526999,
            23.5625451609923,
            9.5955164968327278,
            3.5521600157336479,
            1.0321360914728188,
            0.12414804291083945,
        ];
        let mut dp: Vec<f64> = poles.iter().map(|&p| -p).collect();
        dp.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending, like the C++ dump
        let mut cp: Vec<(f64, f64)> = coeffs
            .iter()
            .zip(poles.iter())
            .map(|(&c, &p)| (c, -p))
            .collect();
        cp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for i in 0..11 {
            // SVD implementation (nalgebra vs LAPACK dgesvd) gives ~1e-8 rel.
            // differences on the weights; poles inherit that scale.
            assert!(
                rel_diff(dp[i], ref_d[i]) < 1e-7,
                "d[{}] = {:.17e} vs C++ {:.17e}",
                i,
                dp[i],
                ref_d[i]
            );
            assert!(
                rel_diff(cp[i].0, ref_c[i]) < 1e-6,
                "c[{}] = {:.17e} vs C++ {:.17e}",
                i,
                cp[i].0,
                ref_c[i]
            );
        }
    }

    #[test]
    fn aaa_alpha020_matches_cpp_reference() {
        // Verification config uses alpha = 1.2 → exponent 0.2 (C++ dump):
        // c[0]=12203.679276445595 d[0]=81810.238449043725 …
        let (coeffs, poles) = compute_partial_fraction_approximation(0.2);
        assert_eq!(poles.len(), 11);
        assert_eq!(coeffs.len(), 11);

        let ref_d = [
            81810.238449043725,
            3758.9223655210712,
            1024.8384999599598,
            384.22885828133542,
            159.41586401409288,
            67.857323058776657,
            28.541088056684991,
            11.499731906580486,
            4.2195395029724878,
            1.2364446030194172,
            0.16695839505676074,
        ];
        let ref_c = [
            12203.679276445595,
            222.67999399384544,
            51.773584135029758,
            19.939594734618325,
            9.290650617314629,
            4.6706524735993993,
            2.4088492159824804,
            1.2461015684184986,
            0.63937694833422642,
            0.31925689946218788,
            0.13670730022888855,
        ];
        let mut dp: Vec<f64> = poles.iter().map(|&p| -p).collect();
        dp.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending, like the C++ dump
        let mut cp: Vec<(f64, f64)> = coeffs
            .iter()
            .zip(poles.iter())
            .map(|(&c, &p)| (c, -p))
            .collect();
        cp.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for i in 0..11 {
            // SVD implementation difference (nalgebra vs LAPACK dgesvd).
            assert!(
                rel_diff(dp[i], ref_d[i]) < 1e-7,
                "d[{}] = {:.17e} vs C++ {:.17e}",
                i,
                dp[i],
                ref_d[i]
            );
            assert!(
                rel_diff(cp[i].0, ref_c[i]) < 1e-6,
                "c[{}] = {:.17e} vs C++ {:.17e}",
                i,
                cp[i].0,
                ref_c[i]
            );
        }
    }

    #[test]
    fn aaa_exact_zero_root_is_dropped() {
        // z=0 is the first support point with f=0 ⇒ the zero polynomial has an
        // exact zero root; the partial-fraction expansion needs the zeros
        // without it (C++: zeros 11 → 10 after DeleteFirst(0.0)).
        // C++ (dggev/LAPACK): alpha=0.5 → 13 support points, 12 poles,
        // 12 coeffs (zeros 12 → 11 after DeleteFirst(0.0)).
        let (coeffs, _poles) = compute_partial_fraction_approximation(0.5);
        assert_eq!(coeffs.len(), 12);
        let (z, f, w) = {
            let lmax = 1000.0_f64;
            let npoints = 1000usize;
            let dx = lmax / (npoints - 1) as f64;
            let x: Vec<f64> = (0..npoints).map(|i| dx * i as f64).collect();
            let val: Vec<f64> = x.iter().map(|&xi| xi.powf(1.0 - 0.5)).collect();
            rational_approximation_aaa(&val, &x, 1e-10, 100)
        };
        assert_eq!(z[0], 0.0, "z=0 must be the first support point");
        assert_eq!(f[0], 0.0, "f(0) = 0");

        let wf: Vec<f64> = w.iter().zip(f.iter()).map(|(&wi, &fi)| wi * fi).collect();
        let zero_poly = weighted_poly_product(&z, &wf);
        assert_eq!(
            zero_poly[0], 0.0,
            "zero polynomial constant term must be exactly 0"
        );
    }
}
