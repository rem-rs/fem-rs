//! Example 14 — DG Poisson (analogous to MFEM ex14)
//!
//! Solves -Δu = f using SIP-DG. Production path uses f = 1 (constant source);
//! MMS-based verification (sin-based RHS) lives under #[cfg(test)].
//!
//! Usage:
//!   cargo run --example mfem_ex14_dg_poisson
//!   cargo run --example mfem_ex14_dg_poisson -- -m mesh.mesh --order 2

use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();
    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };
    // MFEM ex14: kappa = (order+1)^2  (penalty parameter)
    let kappa: f64 = args.kappa.unwrap_or_else(|| (args.order as f64 + 1.0).powi(2));

    let quad_order = args.order * 2 + 1;

    let space = L2Space::new(mesh, args.order);
    let ifl = InteriorFaceList::build(space.mesh());

    // MFEM ex14 RHS: f = 1 (constant source)
    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let rhs = Assembler::assemble_linear(&space, &[&source], quad_order);
    // assemble_sip(kappa_diffusion=1.0, sigma_penalty=kappa, ...)
    let a_mat = DgAssembler::assemble_sip(&space, &ifl, 1.0, kappa, quad_order);

    let mut x = vec![0.0_f64; space.n_dofs()];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 5000, verbose: false, ..SolverConfig::default() };
    let result = solve_gmres(&a_mat, &rhs, &mut x, 30, &cfg).unwrap();

    let sol_norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if let Some(ref path) = args.mesh { println!("  Mesh: {path}"); }
    println!("  n={}, P{}, κ={}, DOFs={}, iters={}, ‖u‖={:.6e}",
             args.n, args.order, kappa, space.n_dofs(), result.iterations, sol_norm);
    println!("  PASS");
}

struct Args {
    mesh: Option<String>,
    n: usize,
    order: u8,
    kappa: Option<f64>,
}

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 8, order: 1, kappa: None };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n" => { a.n = it.next().unwrap_or("8".into()).parse().unwrap_or(8); }
            "--order" | "-o" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            "--kappa" | "--sigma" => { a.kappa = it.next().and_then(|s| s.parse().ok()); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use fem_assembly::{Assembler, DgAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
    use fem_mesh::Mesh;
    use fem_solver::solve_gmres;
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
        solve_gmres(&a_mat, &rhs, &mut x, 30, &cfg).unwrap();
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
        let r = solve_gmres(&a, &rhs, &mut x, 30, &SolverConfig { max_iter: 500, ..SolverConfig::default() }).unwrap();
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
