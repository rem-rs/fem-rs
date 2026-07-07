//! # Example 9 — DG Diffusion (SIP)  (one-to-one with MFEM ex9)
//!
//! Solves the scalar Poisson equation using the Symmetric Interior Penalty
//! Discontinuous Galerkin method with constant source `f = 1`:
//!
//! ```text
//!   −∇·(κ ∇u) = 1    in Ω
//!            u = 0    on ∂Ω  (enforced weakly via penalty)
//! ```
//!
//! ## Usage
//! ```
//! cargo run --example mfem_ex9_dg_advection
//! cargo run --example mfem_ex9_dg_advection -- --n 16 --order 1 --sigma 20
//! cargo run --example mfem_ex9_dg_advection -- -m mesh.mesh --order 1
//! ```

use fem_assembly::{
    Assembler, DgAssembler, InteriorFaceList,
    standard::DomainSourceIntegrator,
};
use fem_io::mfem::read_mfem_file;
use fem_mesh::Mesh;
use fem_solver::{solve_gmres, SolverConfig};
use fem_space::{L2Space, fe_space::FESpace};

struct SolveResult {
    n_elements: usize,
    n_dofs: usize,
    n_interior_faces: usize,
    iterations: usize,
    final_residual: f64,
    converged: bool,
    solution_norm: f64,
}

fn main() {
    let args = parse_args();

    let mesh: Mesh<2> = if let Some(ref path) = args.mesh {
        let mfem = read_mfem_file(path).expect("failed to read MFEM mesh");
        mfem.mesh2d.expect("MFEM mesh must be 2D")
    } else {
        Mesh::<2>::unit_square_tri(args.n)
    };

    println!("=== fem-rs Example 9: SIP-DG Diffusion  (one-to-one with MFEM ex9) ===");
    println!("  Elements: {}, P{} DG", mesh.n_elems(), args.order);
    println!("  Penalty σ = {}", args.sigma);

    let result = solve_case(mesh, args.order, args.sigma);

    let npe = result.n_dofs / result.n_elements;
    let sigma = args.sigma;
    println!("  DOFs: {}  ({} per element)", result.n_dofs, npe);
    println!("  Interior faces: {}", result.n_interior_faces);
    println!("  Effective σ = {:.3}", sigma);
    println!("  Solve: {} iters, residual = {:.3e}, converged = {}",
        result.iterations, result.final_residual, result.converged);
    println!("  ||u_h||_L2 = {:.4e}", result.solution_norm);
    println!("\nDone.");
}

fn solve_case(mesh: Mesh<2>, order: u8, sigma: f64) -> SolveResult {
    let space = L2Space::new(mesh, order);
    let n_dofs = space.n_dofs();

    let ifl = InteriorFaceList::build(space.mesh());
    let kappa = 1.0_f64;
    let mat = DgAssembler::assemble_sip(&space, &ifl, kappa, sigma, order * 2 + 1);

    let source = DomainSourceIntegrator::new(|_: &[f64]| 1.0);
    let rhs = Assembler::assemble_linear(&space, &[&source], order * 2 + 1);

    let mut u = vec![0.0_f64; n_dofs];
    let cfg = SolverConfig { rtol: 1e-10, atol: 0.0, max_iter: 10_000, verbose: false, ..SolverConfig::default() };
    let res = solve_gmres(&mat, &rhs, &mut u, 50, &cfg).expect("DG solve failed");

    let solution_norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();

    SolveResult {
        n_elements: space.mesh().n_elems(),
        n_dofs,
        n_interior_faces: ifl.faces.len(),
        iterations: res.iterations,
        final_residual: res.final_residual,
        converged: res.converged,
        solution_norm,
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

struct Args { mesh: Option<String>, n: usize, order: u8, sigma: f64 }

fn parse_args() -> Args {
    let mut a = Args { mesh: None, n: 16, order: 1, sigma: 4.0 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--mesh" => { a.mesh = it.next(); }
            "--n"     => { a.n     = it.next().unwrap_or("16".into()).parse().unwrap_or(16); }
            "--order" => { a.order = it.next().unwrap_or("1".into()).parse().unwrap_or(1); }
            "--sigma" => { a.sigma = it.next().unwrap_or("4.0".into()).parse().unwrap_or(4.0); }
            _ => {}
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dg_poisson_converges() {
        let r = solve_case(Mesh::<2>::unit_square_tri(8), 1, 20.0);
        assert!(r.converged);
        assert!(r.final_residual < 1.0e-6);
    }
}
