//! ex17_dg_elasticity - baseline vector DG solve using block-diagonal SIP.

use fem_assembly::{Assembler, DgElasticityAssembler, InteriorFaceList, standard::DomainSourceIntegrator};
use fem_mesh::SimplexMesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

fn main() {
    let args = parse_args();

    println!("=== ex17_dg_elasticity (baseline) ===");
    println!("  n={}, order={}, sigma={}", args.n, args.order, args.sigma);

    let mesh = SimplexMesh::<2>::unit_square_tri(args.n);
    let space = L2Space::new(mesh, args.order);
    let ifl = InteriorFaceList::build(space.mesh());
    let n = space.n_dofs();

    let a = DgElasticityAssembler::assemble_sip_vector(&space, &ifl, 1.0, args.sigma, 2, 2 * args.order + 1);

    let fx = DomainSourceIntegrator::new(|_x: &[f64]| 1.0);
    let fy = DomainSourceIntegrator::new(|_x: &[f64]| -1.0);
    let bx = Assembler::assemble_linear(&space, &[&fx], 2 * args.order + 1);
    let by = Assembler::assemble_linear(&space, &[&fy], 2 * args.order + 1);

    let mut b = vec![0.0f64; 2 * n];
    b[..n].copy_from_slice(&bx);
    b[n..].copy_from_slice(&by);

    let mut x = vec![0.0f64; 2 * n];
    let cfg = SolverConfig { rtol: 1e-8, atol: 0.0, max_iter: 5000, verbose: false, ..Default::default() };
    let res = solve_gmres(&a, &b, &mut x, 50, &cfg).expect("GMRES failed");

    println!("  dofs={} (vector={})", n, 2 * n);
    println!("  GMRES iters={}, res={:.3e}, conv={}", res.iterations, res.final_residual, res.converged);
    assert!(res.converged, "DG elasticity baseline solver did not converge");
    println!("  PASS");
}

struct Args {
    n: usize,
    order: u8,
    sigma: f64,
}

fn parse_args() -> Args {
    let mut a = Args { n: 6, order: 1, sigma: 20.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => a.n = it.next().unwrap_or("6".into()).parse().unwrap_or(6),
            "--order" => a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1),
            "--sigma" => a.sigma = it.next().unwrap_or("20.0".into()).parse().unwrap_or(20.0),
            _ => {}
        }
    }
    a
}
