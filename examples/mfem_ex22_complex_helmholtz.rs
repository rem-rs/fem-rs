//! # Example 22 — Complex Time-Harmonic (analogous to MFEM ex22)
//!
//! Solves the damped scalar Helmholtz equation (MFEM ex22, problem type 0):
//!
//! ```text
//!   −∇·(a ∇u) − ω²·b·u + i·ω·c·u = 0    in Ω
//! ```
//!
//! with Dirichlet BCs driving the solution (exact traveling wave imposed on all
//! boundaries).  The exact solution for an "inline-" mesh is
//! `u(x) = exp(-i·κ·x_{dim-1})` where `κ = √(μ·ω·(ε·ω − i·σ))`.
//!
//! ## Usage
//! ```text
//! cargo run --example mfem_ex22_complex_helmholtz
//! cargo run --example mfem_ex22_complex_helmholtz -- --mesh ../data/inline-quad.mesh -o 3
//! cargo run --example mfem_ex22_complex_helmholtz -- --omega 10.0 --sigma 20.0 --mu 1.0
//! ```

use fem_assembly::{
    ComplexAssembler, ComplexGridFunction,
    standard::{DiffusionIntegrator, MassIntegrator},
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_mesh::element_type::ElementType;
use fem_mesh::topology::MeshTopology;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{
    H1Space,
    fe_space::FESpace,
    constraints::boundary_dofs,
};

// ─── Global coefficients (set via CLI) ──────────────────────────────────────
static mut MU: f64 = 1.0;
static mut EPSILON: f64 = 1.0;
static mut SIGMA: f64 = 20.0;
static mut OMEGA: f64 = 10.0;

/// Exact solution: u(x) = exp(-i·κ·x_{d-1})
fn u0_complex(x: &[f64]) -> (f64, f64) {
    let d = x.len();
    let dim = if d >= 1 { d - 1 } else { 0 };
    unsafe {
        let alpha_re = EPSILON * OMEGA;
        let alpha_im = -SIGMA;
        // κ = √(μ·ω·(α_re + i·α_im))
        let tmp_re = MU * OMEGA * alpha_re;
        let tmp_im = MU * OMEGA * alpha_im;
        // sqrt via: √(a+ib) = √((|z|+a)/2) + i·sign(b)·√((|z|-a)/2)
        let mag = (tmp_re * tmp_re + tmp_im * tmp_im).sqrt();
        let kappa_re = ((mag + tmp_re) / 2.0).sqrt();
        let kappa_im = if tmp_im >= 0.0 { 1.0 } else { -1.0 } * ((mag - tmp_re) / 2.0).sqrt();
        // exp(-i·κ·z) = exp(-i·(κ_re+i·κ_im)·z) = exp(κ_im·z)·exp(-i·κ_re·z)
        let z = x[dim];
        let mag_exp = (kappa_im * z).exp();
        let phase = -kappa_re * z;
        (mag_exp * phase.cos(), mag_exp * phase.sin())
    }
}

fn u0_re(x: &[f64]) -> f64 {
    u0_complex(x).0
}
fn u0_im(x: &[f64]) -> f64 {
    u0_complex(x).1
}

fn main() {
    let args = parse_args();
    println!("=== Example 22: Complex Time-Harmonic (MFEM ex22, H1) ===");
    if let Some(ref p) = args.mesh {
        println!("  Mesh file: {p}");
    }
    println!(
        "  μ={:.4}, ε={:.4}, σ={:.4}, ω={:.4}, order={}",
        args.mu, args.eps, args.sigma, args.omega, args.order
    );

    unsafe {
        MU = args.mu;
        EPSILON = args.eps;
        SIGMA = args.sigma;
        OMEGA = args.omega;
    }

    // Load or generate mesh
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    // H1 space
    let space = H1Space::new(mesh, args.order);
    let n = space.n_dofs();
    println!("  DOFs: {n}");

    // Build system: (K − ω²M + i·ω·C) u = 0
    //   K = 1/μ · Diffusion
    //   M = ε · Mass   → ω²·ε = 1/μ · ω²·μ·ε (but MFEM convention)
    //   C = σ · Mass
    let stiff_coef = 1.0 / args.mu;
    let mass_coef = args.eps; // M = ε·M_base; assembler computes -ω²·M = -ω²·ε·M_base
    let damp_coef = args.sigma; // C = σ·M; assembler then computes k_im = ω·C = ω·σ·M

    let mut sys = ComplexAssembler::assemble(
        &space,
        &[&DiffusionIntegrator {
            kappa: stiff_coef,
        }],
        &[&MassIntegrator { rho: mass_coef }],
        &[&MassIntegrator { rho: damp_coef }],
        args.omega,
        args.order * 2 + 1,
    );

    // Project exact solution for BCs
    let dm = space.dof_manager();
    let bnd: Vec<usize> = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags())
        .into_iter()
        .map(|d| d as usize)
        .collect();

    let exact_re: Vec<f64> = bnd
        .iter()
        .map(|&d| {
            let coord = space.dof_manager().dof_coord(d as u32);
            u0_re(&coord)
        })
        .collect();
    let exact_im: Vec<f64> = bnd
        .iter()
        .map(|&d| {
            let coord = space.dof_manager().dof_coord(d as u32);
            u0_im(&coord)
        })
        .collect();

    let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);
    sys.apply_dirichlet(&bnd, &exact_re, &exact_im, &mut rhs);

    // Solve
    let a = sys.to_flat_csr();
    let mut x = vec![0.0; 2 * n];
    let cfg = SolverConfig {
        rtol: 1e-8,
        atol: 1e-14,
        max_iter: 3000,
        verbose: false,
        ..SolverConfig::default()
    };
    let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES did not converge");

    let gf = ComplexGridFunction::from_flat(&x);
    let amp = gf.amplitude();
    let max_amp = amp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_amp = amp.iter().cloned().fold(f64::INFINITY, f64::min);

    println!(
        "  GMRES: {} iters, residual={:.3e}, converged={}",
        res.iterations, res.final_residual, res.converged
    );
    println!("  |u| ∈ [{:.6e}, {:.6e}]", min_amp, max_amp);

    // L2 error vs exact via element quadrature (matches C++ ComputeL2Error)
    let (err_re, err_im, norm_re, norm_im) = compute_complex_l2_error(
        &space, &gf.u_re, &gf.u_im, u0_re, u0_im, args.order * 2 + 1,
    );
    println!(
        "  L2 error (abs): re={:.6e}, im={:.6e}  (rel: re={:.3e}, im={:.3e})",
        err_re, err_im, err_re / norm_re.max(1e-14), err_im / norm_im.max(1e-14)
    );

    assert!(res.converged, "GMRES did not converge");
    println!("  PASS");
}

// ─── CLI ────────────────────────────────────────────────────────────────────

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    mu: f64,
    eps: f64,
    sigma: f64,
    omega: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        n: 8,
        order: 1,
        mu: 1.0,
        eps: 1.0,
        sigma: 20.0,
        omega: 10.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => a.mesh = it.next(),
            "--n" => {
                a.n = it
                    .next()
                    .unwrap_or("8".into())
                    .parse()
                    .unwrap_or(8)
            }
            "-o" | "--order" => {
                a.order = it
                    .next()
                    .unwrap_or("1".into())
                    .parse()
                    .unwrap_or(1)
            }
            "--mu" => {
                a.mu = it
                    .next()
                    .unwrap_or("1.0".into())
                    .parse()
                    .unwrap_or(1.0)
            }
            "--eps" => {
                a.eps = it
                    .next()
                    .unwrap_or("1.0".into())
                    .parse()
                    .unwrap_or(1.0)
            }
            "--sigma" => {
                a.sigma = it
                    .next()
                    .unwrap_or("20.0".into())
                    .parse()
                    .unwrap_or(20.0)
            }
            "--omega" => {
                a.omega = it
                    .next()
                    .unwrap_or("10.0".into())
                    .parse()
                    .unwrap_or(10.0)
            }
            _ => {}
        }
    }
    a
}

/// Compute L² error using element quadrature (matches C++ ComputeL2Error).
///
/// For each element, evaluates u_h and u_exact at quadrature points and
/// integrates `∫(u_h − u_exact)² dx`.
fn compute_complex_l2_error<S: FESpace>(
    space: &S,
    u_re: &[f64],
    u_im: &[f64],
    exact_re: impl Fn(&[f64]) -> f64,
    exact_im: impl Fn(&[f64]) -> f64,
    quad_order: u8,
) -> (f64, f64, f64, f64) {
    use fem_element::lagrange::{TriP1, TriP2, TriP3, TetP1, TetP2, TetP3};
    use fem_element::ReferenceElement;

    let mesh = space.mesh();
    let dim = mesh.dim() as usize;
    let order = space.order();

    let ref_elem_vol = |et: ElementType, order: u8| -> Box<dyn ReferenceElement> {
        match (et, order) {
            (ElementType::Tri3, 1) => Box::new(TriP1),
            (ElementType::Tri3, 2) => Box::new(TriP2),
            (ElementType::Tri3, 3) => Box::new(TriP3),
            (ElementType::Tet4, 1) => Box::new(TetP1),
            (ElementType::Tet4, 2) => Box::new(TetP2),
            (ElementType::Tet4, 3) => Box::new(TetP3),
            _ => panic!("unsupported element for L2 error: {et:?} o={order}"),
        }
    };

    let mut err_re2 = 0.0_f64;
    let mut err_im2 = 0.0_f64;
    let mut norm_re2 = 0.0_f64;
    let mut norm_im2 = 0.0_f64;

    for e in mesh.elem_iter() {
        let elem_type = mesh.element_type(e);
        let ref_elem = ref_elem_vol(elem_type, order);
        let n_ldofs = ref_elem.n_dofs();
        let elem_dofs: Vec<usize> = space.element_dofs(e).iter().map(|&d| d as usize).collect();
        let nodes = mesh.element_nodes(e);

        // Jacobian determinant for reference→physical mapping
        let x0: Vec<f64> = mesh.node_coords(nodes[0]).to_vec();
        let x1: Vec<f64> = mesh.node_coords(nodes[1]).to_vec();
        let det_j = if dim == 2 {
            let x2 = mesh.node_coords(nodes[2]);
            (x1[0] - x0[0]) * (x2[1] - x0[1]) - (x1[1] - x0[1]) * (x2[0] - x0[0])
        } else {
            let x2 = mesh.node_coords(nodes[2]);
            let x3 = mesh.node_coords(nodes[3]);
            let j00 = x1[0] - x0[0]; let j01 = x2[0] - x0[0]; let j02 = x3[0] - x0[0];
            let j10 = x1[1] - x0[1]; let j11 = x2[1] - x0[1]; let j12 = x3[1] - x0[1];
            let j20 = x1[2] - x0[2]; let j21 = x2[2] - x0[2]; let j22 = x3[2] - x0[2];
            j00 * (j11 * j22 - j12 * j21) - j01 * (j10 * j22 - j12 * j20) + j02 * (j10 * j21 - j11 * j20)
        };

        let quad = ref_elem.quadrature(quad_order);
        let mut phi = vec![0.0_f64; n_ldofs];

        // Physical coordinate mapping: x_phys = x0 + J * xi
        let j_cols: Vec<Vec<f64>> = (0..dim)
            .map(|k| {
                let nk = mesh.node_coords(nodes[k + 1]);
                (0..dim).map(|d| nk[d] - x0[d]).collect()
            })
            .collect();

        for (qi, xi) in quad.points.iter().enumerate() {
            let w = quad.weights[qi] * det_j.abs();

            // Physical coordinates: x0 + Σ_k J_col[k] * xi[k]
            let mut x_phys: Vec<f64> = x0.clone();
            for k in 0..dim {
                for d in 0..dim { x_phys[d] += j_cols[k][d] * xi[k]; }
            }

            // Evaluate u_h at this quadrature point
            ref_elem.eval_basis(xi, &mut phi);
            let mut uh_re = 0.0;
            let mut uh_im = 0.0;
            for i in 0..n_ldofs {
                uh_re += phi[i] * u_re[elem_dofs[i]];
                uh_im += phi[i] * u_im[elem_dofs[i]];
            }

            // Exact solution
            let ue_re = exact_re(&x_phys);
            let ue_im = exact_im(&x_phys);

            let dr = uh_re - ue_re;
            let di = uh_im - ue_im;
            err_re2 += w * dr * dr;
            err_im2 += w * di * di;
            norm_re2 += w * ue_re * ue_re;
            norm_im2 += w * ue_im * ue_im;
        }
    }
    (err_re2.sqrt(), err_im2.sqrt(), norm_re2.sqrt(), norm_im2.sqrt())
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn base_args() -> Args {
        Args {
            mesh: None,
            n: 10,
            order: 1,
            mu: 1.0,
            eps: 1.0,
            sigma: 20.0,
            omega: 10.0,
        }
    }

    #[test]
    fn ex22_h1_converges_with_reasonable_error() {
        let args = base_args();
        unsafe {
            MU = args.mu;
            EPSILON = args.eps;
            SIGMA = args.sigma;
            OMEGA = args.omega;
        }
        let mesh = Mesh::<2>::unit_square_tri(args.n);
        let space = H1Space::new(mesh, args.order);
        let mut sys = ComplexAssembler::assemble(
            &space,
            &[&DiffusionIntegrator {
                kappa: 1.0 / args.mu,
            }],
            &[&MassIntegrator {
                rho: -args.omega * args.omega * args.eps,
            }],
            &[&MassIntegrator {
                rho: args.omega * args.sigma,
            }],
            args.omega,
            args.order * 2 + 1,
        );
        let n = space.n_dofs();
        let dm = space.dof_manager();
        let bnd: Vec<usize> = boundary_dofs(space.mesh(), dm, &space.mesh().unique_boundary_tags())
            .into_iter()
            .map(|d| d as usize)
            .collect();
        let exact_re: Vec<f64> = bnd
            .iter()
            .map(|&d| {
                let coord = space.dof_manager().dof_coord(d as u32);
                u0_re(&coord)
            })
            .collect();
        let exact_im: Vec<f64> = bnd
            .iter()
            .map(|&d| {
                let coord = space.dof_manager().dof_coord(d as u32);
                u0_im(&coord)
            })
            .collect();
        let mut rhs = sys.assemble_rhs(&vec![0.0; n], &vec![0.0; n]);
        sys.apply_dirichlet(&bnd, &exact_re, &exact_im, &mut rhs);
        let a = sys.to_flat_csr();
        let mut x = vec![0.0; 2 * n];
        let cfg = SolverConfig {
            rtol: 1e-8,
            atol: 1e-14,
            max_iter: 3000,
            verbose: false,
            ..SolverConfig::default()
        };
        let res = solve_gmres(&a, &rhs, &mut x, 50, &cfg).expect("GMRES did not converge");
        assert!(res.converged);
        assert!(res.final_residual < 1.0e-6, "residual = {}", res.final_residual);
    }

    #[test]
    fn ex22_dof_count_matches_p1_h1_formula() {
        for &n in &[6usize, 10usize] {
            let mut a = base_args();
            a.n = n;
            unsafe {
                MU = a.mu;
                EPSILON = a.eps;
                SIGMA = a.sigma;
                OMEGA = a.omega;
            }
            let mesh = Mesh::<2>::unit_square_tri(n);
            let space = H1Space::new(mesh, 1);
            assert_eq!(space.n_dofs(), (n + 1) * (n + 1));
        }
    }
}
