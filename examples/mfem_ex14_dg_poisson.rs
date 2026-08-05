//! Example 14 — DG Poisson (1:1 translation of MFEM ex14)
//!
//! Solves −Δu = 1 with homogeneous Dirichlet BCs using SIP-DG on the L² space:
//!
//! ```text
//!   a = ∫ κ ∇u·∇v + interior/boundary face terms of DGDiffusionIntegrator(κ, σ, k)
//!   b = ∫ 1·v  (the DGDirichletLFIntegrator term is zero for homogeneous BCs)
//! ```
//!
//! Solver: sigma == −1 → PCG + GSSmoother; otherwise GMRES(10) + GSSmoother
//! (matching MFEM ex14: `PCG(A, M, b, x, 1, 500, 1e-12, 0.0)` / `GMRES(..., 10, ...)`).
//!
//! Usage:
//! ```text
//! cargo run --example mfem_ex14_dg_poisson
//! cargo run --example mfem_ex14_dg_poisson -- -m ../data/star.mesh -r 4 -o 2
//! ```

use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_mesh::{refine_uniform, Mesh};
use fem_solver::{solve_gmres_gssmoother, solve_pcg_gssmoother, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    let dim = 2usize;

    let mut mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        let mfem = read_mfem_file("data/star.mesh").expect("failed to read data/star.mesh");
        mfem.mesh2d.expect("star.mesh must be a 2-D mesh")
    };

    // Auto refinement level (MFEM ex14): floor(log(50000/NE)/log(2)/dim), -1 for auto.
    let ref_levels = if args.ref_levels < 0 {
        ((50000.0_f64 / mesh.n_elems() as f64).ln() / 2.0_f64.ln() / dim as f64).floor()
            as i32
    } else {
        args.ref_levels
    };
    for _ in 0..ref_levels {
        mesh = refine_uniform(&mesh);
    }

    // MFEM ex14: kappa = (order+1)^2 when negative (penalty parameter).
    let kappa: f64 = if args.kappa < 0.0 {
        (args.order as f64 + 1.0).powi(2)
    } else {
        args.kappa
    };

    let quad_order = args.order * 2 + 1;

    let space = L2Space::new(mesh, args.order);
    let ifl = InteriorFaceList::build(space.mesh());
    println!("Number of unknowns: {}", space.n_dofs());

    // MFEM ex14 RHS: f = 1 (constant source).  The boundary LF term
    // DGDirichletLFIntegrator(zero, one, sigma, kappa) is identically zero
    // because the Dirichlet data uD = 0.
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let rhs = Assembler::assemble_linear(&space, &[&source], quad_order);
    // DGDiffusionIntegrator(one, sigma, kappa): volume diffusion + interior and
    // boundary face penalty (weak Dirichlet).
    let a_mat = DgAssembler::assemble_dg(&space, &ifl, 1.0, args.sigma, kappa, quad_order, None);

    let mut x = vec![0.0_f64; space.n_dofs()];
    let cfg = SolverConfig {
        rtol: 1e-12,
        atol: 0.0,
        max_iter: 500,
        verbose: true,
        ..SolverConfig::default()
    };
    if args.sigma == -1.0 {
        let _ = solve_pcg_gssmoother(&a_mat, &rhs, &mut x, &cfg).expect("PCG failed");
    } else {
        let _ = solve_gmres_gssmoother(&a_mat, &rhs, &mut x, 10, &cfg).expect("GMRES failed");
    }

    // MFEM ex14 outputs: refined.mesh (precision 8) and sol.gf (precision 8).
    fem_io::mfem::write_mfem_file("refined.mesh", space.mesh()).ok();
    fem_io::mfem::write_mfem_gf_file("sol.gf", dim, &x, "DG", args.order, 1, 8).ok();
}

struct Args {
    mesh: Option<String>,
    ref_levels: i32,
    order: u8,
    sigma: f64,
    kappa: f64,
}

fn parse_args() -> Args {
    let mut a = Args {
        mesh: None,
        ref_levels: -1,
        order: 1,
        sigma: -1.0,
        kappa: -1.0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "-r" | "--refine" => {
                a.ref_levels = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1);
            }
            "-o" | "--order" => {
                a.order = it.next().and_then(|s| s.parse().ok()).unwrap_or(1);
            }
            "-s" | "--sigma" => {
                a.sigma = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "-k" | "--kappa" => {
                a.kappa = it.next().and_then(|s| s.parse().ok()).unwrap_or(-1.0);
            }
            "-e" | "--eta" => {
                // BR2 penalty (eta > 0) is not implemented; ex14 default is 0.
                let eta: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
                if eta > 0.0 {
                    panic!("mfem_ex14_dg_poisson: BR2 (eta > 0) is not implemented");
                }
            }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
    use fem_mesh::Mesh;
    use fem_solver::solve_cg;
    use fem_solver::SolverConfig;
    use fem_space::L2Space;
    use fem_space::fe_space::FESpace;

    fn solve_dg_poisson_mms(n: usize, order: u8) -> f64 {
        // MFEM ex14: kappa = (order+1)^2
        let kappa: f64 = (order as f64 + 1.0).powi(2);
        let mesh = Mesh::<2>::unit_square_tri(n);
        let space = L2Space::new(mesh, order);
        let ifl = InteriorFaceList::build(space.mesh());

        // MMS RHS: f = 2π² sin(πx) sin(πy) for u = sin(πx) sin(πy)
        let source = DomainSourceIntegrator::new(|x: &[f64]| {
            let p = std::f64::consts::PI;
            2.0 * p * p * (p * x[0]).sin() * (p * x[1]).sin()
        });
        let rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);
        let a_mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, kappa, order * 2 + 1);

        let mut x = vec![0.0_f64; space.n_dofs()];
        let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
        solve_cg(&a_mat, &rhs, &mut x, &cfg).unwrap();
        x.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    #[test]
    fn smoke() {
        let mesh = Mesh::<2>::unit_square_tri(4);
        let space = L2Space::new(mesh, 1);
        let ifl = InteriorFaceList::build(space.mesh());
        let rhs = Assembler::assemble_linear(&space, &[&DomainSourceIntegrator::new(|_| 1.0)], 3);
        let a = DgAssembler::assemble_sip(&space, &ifl, 1.0, 4.0, 3);
        let mut x = vec![0.0; space.n_dofs()];
        let r = solve_cg(&a, &rhs, &mut x, &SolverConfig { max_iter: 500, ..SolverConfig::default() }).unwrap();
        assert!(r.iterations > 0);
    }

    #[test]
    fn regression_solution_norm() {
        let norm = solve_dg_poisson_mms(8, 1);
        fem_regression::regression("mfem_ex14_dg_poisson")
            .check("sol_norm_n8_p1", norm)
            .finalize();
    }
}
